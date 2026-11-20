// Reproducer: concurrent dma-buf import (zeMemAllocDevice + ze_external_memory_import_fd_t)
// and kernel submit with indirect access. See README.md.
#include <level_zero/ze_api.h>

#include <drm/xe_drm.h>
#include <fcntl.h>
#include <getopt.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

// Any Level Zero error ends the run through fail() (exit code 1).
#define CHECK(call)                                                                         \
    do {                                                                                    \
        ze_result_t r_ = (call);                                                            \
        if (r_ != ZE_RESULT_SUCCESS) fail(#call, r_);                                       \
    } while (0)

static constexpr uint8_t kPattern = 0xA5;
static constexpr uint32_t kStride = 4096; // one read per page
static std::atomic<uint64_t> g_done{0};   // imports finished over all threads

// Import currently in use by each worker (GPU address and size), for the fault summary.
struct LiveImport {
    std::atomic<uint64_t> addr{0}, size{0};
};
static std::unique_ptr<LiveImport[]> g_live;
static int g_liveCount = 0;
static int g_kmsg = -1; // /dev/kmsg, positioned at its end when the program started

// One line: where the xe page faults logged since start lie relative to the imports in flight.
static void reportFaults() {
    if (g_kmsg < 0) {
        fprintf(stderr, "page faults: /dev/kmsg not readable (run as root to see them)\n");
        return;
    }
    usleep(200 * 1000); // let the kernel finish logging
    // collect the distinct faulted addresses from the xe fault dumps
    std::set<uint64_t> faulted;
    char rec[8192];
    ssize_t len;
    while ((len = read(g_kmsg, rec, sizeof(rec) - 1)) > 0) {
        rec[len] = 0;
        for (const char *p = rec; (p = strstr(p, "Faulted Address: 0x")); p += 1)
            faulted.insert(strtoull(p + strlen("Faulted Address: 0x"), nullptr, 16));
    }
    // count the faults that land inside an import some worker is using
    size_t inside = 0;
    for (uint64_t a : faulted)
        for (int t = 0; t < g_liveCount; t++) {
            // the kernel logs 48-bit addresses, Level Zero returns them sign-extended
            uint64_t base = g_live[t].addr.load() & 0xffffffffffffull;
            if (base && a >= base && a < base + g_live[t].size.load()) { inside++; break; }
        }
    if (faulted.empty())
        fprintf(stderr, "page faults: none logged\n");
    else
        fprintf(stderr, "page faults: %zu distinct addresses, %zu inside imports in flight, %zu outside\n",
                faulted.size(), inside, faulted.size() - inside);
}

// Report the failure and exit with `code`. _exit skips destructors and Level Zero teardown,
// which can hang or crash once the device is lost.
[[noreturn]] static void fail(const char *what, int result, int code = 1) {
    // all threads fail together once the device is lost, only the first one reports;
    // the others park here until _exit
    static std::atomic<bool> reported{false};
    if (reported.exchange(true))
        for (;;) pause();
    fflush(stdout); // _exit drops buffered stdout, e.g. the driver version line when piped
    fprintf(stderr, "FAIL after %lu imports: %s -> 0x%x\n", (unsigned long)g_done.load(), what, result);
    reportFaults();
    fflush(stderr);
    _exit(code);
}

// A dma-buf made outside Level Zero: system memory BO from the xe kernel driver, exported
// as an fd and filled with kPattern through an mmap of the fd. It must be made on the same GPU
// as the Level Zero device under test (see --render-node).
static int makeDmaBuf(const char *renderNode, size_t size) {
    int drm = open(renderNode, O_RDWR);
    if (drm < 0) { perror(renderNode); exit(2); }
    drm_xe_gem_create create{};
    create.size = size;
    create.placement = 1u << 0; // memory region 0, which is system memory on xe
    create.cpu_caching = DRM_XE_GEM_CPU_CACHING_WB; // WB is not allowed for VRAM placements
    if (ioctl(drm, DRM_IOCTL_XE_GEM_CREATE, &create)) { perror("DRM_IOCTL_XE_GEM_CREATE (xe driver?)"); exit(2); }
    drm_prime_handle prime{};
    prime.handle = create.handle;
    prime.flags = DRM_CLOEXEC | DRM_RDWR;
    if (ioctl(drm, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime)) { perror("PRIME_HANDLE_TO_FD"); exit(2); }
    void *map = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, prime.fd, 0);
    if (map == MAP_FAILED) { perror("mmap dma-buf"); exit(2); }
    memset(map, kPattern, size);
    munmap(map, size);
    return prime.fd; // the drm fd and GEM handle stay open for the life of the process
}

// Read-only state shared by all workers.
struct Shared {
    ze_context_handle_t ctx;
    ze_device_handle_t dev;
    ze_module_handle_t mod;
    int dmabufFd;
    size_t size;
    int iterations;
    bool makeResident; // workaround: bind each import explicitly right after it is created
    bool importsOnly;  // benchmark: import and free only, no kernel (no residency scan, cannot fail)
};

// One thread: import the dma-buf, read it with a kernel, check the data, free it; repeat.
// Each thread has its own kernel (argument state is per kernel) and command list.
static void worker(Shared *s, int index) {
    // Indirect access makes every launch of this kernel run the driver's indirect-allocation
    // residency scan (makeIndirectAllocationsResident), which is the side of the race that
    // skips the import.
    ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC};
    kd.pKernelName = "read_through_table";
    ze_kernel_handle_t kernel;
    CHECK(zeKernelCreate(s->mod, &kd, &kernel));
    CHECK(zeKernelSetIndirectAccess(kernel, ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE));

    // One work-item per page, 64 per group, so one group covers 256 KiB. The group count below
    // rounds down and the kernel has no bounds check, which is why --size-kib must be a multiple
    // of 256: otherwise the tail pages are never read and the data check fails spuriously.
    const uint32_t pages = s->size / kStride, group = 64;
    CHECK(zeKernelSetGroupSize(kernel, group, 1, 1));

    // table holds the import's address, so the kernel reaches the import only through memory,
    // never as an argument; out gets one byte per page
    ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    uint64_t *table;
    uint8_t *out;
    CHECK(zeMemAllocHost(s->ctx, &hd, sizeof(uint64_t), 64, (void **)&table));
    CHECK(zeMemAllocHost(s->ctx, &hd, pages, 64, (void **)&out));
    uint32_t stride = kStride;
    CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(table), &table));
    CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(out), &out));
    CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(stride), &stride));

    // immediate + synchronous: every append submits at once, giving many residency scans
    ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    qd.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    ze_command_list_handle_t list;
    CHECK(zeCommandListCreateImmediate(s->ctx, s->dev, &qd, &list));
    ze_group_count_t groups{pages / group, 1, 1};

    for (int i = 0; i < s->iterations; i++) {
        // import: the driver takes ownership of the fd it is given, hence the dup
        ze_external_memory_import_fd_t imp{ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD};
        imp.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
        imp.fd = dup(s->dmabufFd);
        ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
        dd.pNext = &imp;
        void *imported;
        CHECK(zeMemAllocDevice(s->ctx, &dd, s->size, 0, s->dev, &imported));
        // Workaround used by torchcodec-xpu: residency by pointer lookup does not depend on the
        // allocation id, so a skipped indirect-allocation scan no longer leaves the import unbound.
        if (s->makeResident) CHECK(zeContextMakeMemoryResident(s->ctx, s->dev, imported, s->size));

        if (s->importsOnly) {
            CHECK(zeMemFree(s->ctx, imported));
            g_done++;
            continue;
        }

        // use it immediately, as a decoder would; publish it first for the fault summary
        g_live[index].size = s->size;
        g_live[index].addr = (uint64_t)imported;
        *table = (uint64_t)imported;
        memset(out, 0, pages); // so a page the kernel did not read shows up as wrong data
        ze_result_t r = zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0, nullptr);
        if (r == ZE_RESULT_SUCCESS)
            r = zeCommandListHostSynchronize(list, 20ull * 1000 * 1000 * 1000); // ns
        if (r == ZE_RESULT_NOT_READY) fail("kernel did not finish in 20 s (hang)", r, 3);
        if (r != ZE_RESULT_SUCCESS) fail("kernel launch/sync", r);
        for (uint32_t p = 0; p < pages; p++)
            if (out[p] != kPattern) { fprintf(stderr, "WRONG DATA page %u = 0x%02x\n", p, out[p]); fail("data check", 0, 4); }

        // unpublish before the free, so the address is not counted once it can be reused
        g_live[index].addr = 0;
        CHECK(zeMemFree(s->ctx, imported));
        g_done++;
    }
}

static void usage(FILE *out, const char *prog) {
    fprintf(out,
            "Usage: %s [options]\n"
            "\n"
            "Imports a dma-buf from several threads while each thread submits a kernel with\n"
            "indirect access that reads the import. Exits 0 on pass, non-zero on failure.\n"
            "\n"
            "Options:\n"
            "  -k, --kernel FILE       SPIR-V kernel built by build.sh (default: kernel.spv)\n"
            "  -n, --render-node PATH  render node of the xe GPU (default: first node driven by xe)\n"
            "  -d, --device N          Level Zero device index (default: 0)\n"
            "  -t, --threads N         worker threads (default: 8)\n"
            "  -i, --imports N         imports per thread (default: 2000)\n"
            "  -s, --size-kib N        buffer size in KiB, multiple of 256 (default: 3072)\n"
            "  -r, --make-resident     workaround: zeContextMakeMemoryResident after each import\n"
            "  -o, --imports-only      benchmark: import and free only, no kernel (cannot fail)\n"
            "  -h, --help              show this help\n"
            "\n"
            "Exit codes: 0 pass, 1 Level Zero error (device lost), 2 setup error or bad usage,\n"
            "3 kernel hang, 4 wrong data.\n",
            prog);
}

// Parse an integer option; exits with usage on error. Only --size-kib passes multipleOf,
// so the message says "multiple of 256" for any multipleOf > 1.
static long parseLong(const char *prog, const char *opt, const char *text, long min, long multipleOf = 1) {
    char *end;
    errno = 0;
    long v = strtol(text, &end, 10);
    if (errno || *end || v < min || v % multipleOf) {
        fprintf(stderr, "%s: invalid value '%s' for %s (integer >= %ld%s expected)\n", prog, text, opt, min,
                multipleOf > 1 ? ", multiple of 256" : "");
        usage(stderr, prog);
        exit(2);
    }
    return v;
}

// First render node (minors 128-191) driven by the xe kernel driver, "" if there is none.
static std::string findXeRenderNode() {
    for (int minor = 128; minor < 192; minor++) {
        char link[128], target[256];
        snprintf(link, sizeof(link), "/sys/class/drm/renderD%d/device/driver", minor);
        ssize_t len = readlink(link, target, sizeof(target) - 1);
        if (len < 0) continue;
        target[len] = 0;
        const char *base = strrchr(target, '/');
        if (!strcmp(base ? base + 1 : target, "xe")) return "/dev/dri/renderD" + std::to_string(minor);
    }
    return "";
}

int main(int argc, char **argv) {
    const char *spv = "kernel.spv";
    std::string nodeArg;
    int threads = 8, iterations = 2000, devIndex = 0;
    size_t size = 3072 * 1024;
    bool makeResident = false, importsOnly = false;

    // command line
    static const option longOptions[] = {
        {"kernel", required_argument, nullptr, 'k'},      {"render-node", required_argument, nullptr, 'n'},
        {"device", required_argument, nullptr, 'd'},      {"threads", required_argument, nullptr, 't'},
        {"imports", required_argument, nullptr, 'i'},     {"size-kib", required_argument, nullptr, 's'},
        {"make-resident", no_argument, nullptr, 'r'},     {"imports-only", no_argument, nullptr, 'o'},
        {"help", no_argument, nullptr, 'h'},              {nullptr, 0, nullptr, 0}};
    int c;
    while ((c = getopt_long(argc, argv, "k:n:d:t:i:s:roh", longOptions, nullptr)) != -1) {
        switch (c) {
        case 'k': spv = optarg; break;
        case 'n': nodeArg = optarg; break;
        case 'd': devIndex = (int)parseLong(argv[0], "--device", optarg, 0); break;
        case 't': threads = (int)parseLong(argv[0], "--threads", optarg, 1); break;
        case 'i': iterations = (int)parseLong(argv[0], "--imports", optarg, 1); break;
        case 's': size = (size_t)parseLong(argv[0], "--size-kib", optarg, 256, 256) * 1024; break;
        case 'r': makeResident = true; break;
        case 'o': importsOnly = true; break;
        case 'h': usage(stdout, argv[0]); return 0;
        default: usage(stderr, argv[0]); return 2;
        }
    }
    if (optind != argc) {
        fprintf(stderr, "%s: unexpected argument '%s'\n", argv[0], argv[optind]);
        usage(stderr, argv[0]);
        return 2;
    }
    std::string node = nodeArg.empty() ? findXeRenderNode() : nodeArg;
    if (node.empty()) {
        fprintf(stderr, "%s: no render node driven by xe found, pass --render-node\n", argv[0]);
        return 2;
    }
    const char *nodePath = node.c_str();

    // fault summary state; seek /dev/kmsg to its end so only faults from this run are counted
    g_live.reset(new LiveImport[threads]);
    g_liveCount = threads;
    g_kmsg = open("/dev/kmsg", O_RDONLY | O_NONBLOCK);
    if (g_kmsg >= 0) lseek(g_kmsg, 0, SEEK_END);

    // Level Zero setup: first driver, chosen device
    CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t n = 1;
    ze_driver_handle_t drv;
    CHECK(zeDriverGet(&n, &drv));
    uint32_t nd = 0;
    CHECK(zeDeviceGet(drv, &nd, nullptr));
    std::vector<ze_device_handle_t> devs(nd);
    CHECK(zeDeviceGet(drv, &nd, devs.data()));
    ze_driver_properties_t dp{ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES};
    zeDriverGetProperties(drv, &dp);
    ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    if ((uint32_t)devIndex >= nd) {
        fprintf(stderr, "%s: --device %d, but there are only %u Level Zero devices\n", argv[0], devIndex, nd);
        return 2;
    }
    zeDeviceGetProperties(devs[devIndex], &props);
    printf("driver version 0x%x, device %d of %u: %s, %d threads x %d imports, %zu KiB%s%s\n", dp.driverVersion,
           devIndex, nd, props.name, threads, iterations, size / 1024,
           makeResident ? ", zeContextMakeMemoryResident after each import" : "",
           importsOnly ? ", imports only (no kernel)" : "");

    // one context and one module shared by all threads
    ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t ctx;
    CHECK(zeContextCreate(drv, &cd, &ctx));
    std::ifstream f(spv, std::ios::binary);
    std::vector<char> il((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (il.empty()) { fprintf(stderr, "cannot read %s\n", spv); return 2; }
    ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC};
    md.format = ZE_MODULE_FORMAT_IL_SPIRV;
    md.inputSize = il.size();
    md.pInputModule = (const uint8_t *)il.data();
    ze_module_handle_t mod;
    CHECK(zeModuleCreate(ctx, devs[devIndex], &md, &mod, nullptr));

    // one dma-buf, imported again and again by every thread
    Shared s{ctx, devs[devIndex], mod, makeDmaBuf(nodePath, size), size, iterations, makeResident, importsOnly};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(worker, &s, t);
    for (auto &t : pool) t.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("PASS: %lu imports, %.3f s, %.0f imports/s\n", (unsigned long)g_done.load(), secs, g_done.load() / secs);
    return 0;
}
