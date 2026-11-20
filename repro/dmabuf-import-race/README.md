# Level Zero: concurrent dma-buf import leaves the allocation unbound (device lost)

Standalone reproducer for a race between a dma-buf import and the indirect-access residency scan.
`MemoryManager::importFdHandle` takes the allocation id with `++allocationsCounter`
([memory_manager.cpp#L1797](https://github.com/intel/compute-runtime/blob/38c85b78896bf7bd2c1a2efa9adc81735f5d6ae9/shared/source/memory_manager/memory_manager.cpp#L1797)),
and its caller `Context::getMemHandlePtr` only later publishes the allocation in
`internalAllocationsMap` with `insertSVMAlloc`
([context.cpp#L931](https://github.com/intel/compute-runtime/blob/38c85b78896bf7bd2c1a2efa9adc81735f5d6ae9/level_zero/core/source/context/context.cpp#L931);
links at master `38c85b7889`. Before `f2ce53c299` the insert was at the end of `importFdHandle`).
`SVMAllocsManager::makeIndirectAllocationsResident` (run when a command list submits a kernel with
indirect access) scans ids up to the counter and then advances `latestResidentObjectId`. If that scan
runs between the id bump and the publish, the id is skipped for good: no `DRM_XE_VM_BIND` is issued
for the import, and a kernel reading it page-faults (`ZE_RESULT_ERROR_DEVICE_LOST`).

## What the program does

1. Creates a 3 MiB system-memory buffer object directly with the xe driver
   (`DRM_IOCTL_XE_GEM_CREATE`), exports it as a dma-buf and fills it with `0xA5`. Level Zero is not
   involved in making the buffer, so the driver under test only sees it as an imported fd.
2. Starts N threads. Each has its own immediate command list and kernel, and loops: `dup` the fd,
   import it with `zeMemAllocDevice` + `ze_external_memory_import_fd_t`, launch a kernel that reads
   one byte per page of the import, wait, check every byte is `0xA5`, `zeMemFree`.
3. The kernel (`kernel.cl`) gets the imported pointer through a table in host memory, not as an
   argument, and the kernel sets `zeKernelSetIndirectAccess(DEVICE)`. That makes the buffer
   reachable only through the indirect-access residency scan, as in the real workload (a video
   decoder importing VA-API frames while other threads run SYCL kernels).
4. The kernel reads one byte per 4 KiB page, which touches every page of the import while keeping
   the kernel cheap. It runs one work-item per page in work-groups of 64, so one group covers
   64 x 4 KiB = 256 KiB. The group count is `pages / 64`, rounded down, and the kernel has no bounds
   check, so the buffer size must be a multiple of 256 KiB (`--size-kib` enforces this). With any
   other size the last pages would never be read, their result bytes would stay `0` and the data
   check would fail without a driver bug.

A failing run exits non-zero and prints the number of imports done and the Level Zero result.

## Build and run

Needs `g++`, Level Zero headers and loader, `ocloc`, and `drm/xe_drm.h`.

```
./build.sh
./repro
```

This runs 8 threads x 2000 imports of a 3 MiB buffer on the first Level Zero device, using the
first render node driven by `xe`. It prints `PASS` and exits 0, or prints the failing call and the
Level Zero result (`0x70000001` is device lost) and exits non-zero.

## Additional options

| Option | Short | Default | Meaning |
|---|---|---|---|
| `--kernel FILE` | `-k` | `kernel.spv` | SPIR-V kernel built by `build.sh` |
| `--render-node PATH` | `-n` | first `xe` node | Render node used to create the dma-buf. Must be the same GPU as the Level Zero device |
| `--device N` | `-d` | `0` | Level Zero device index |
| `--threads N` | `-t` | `8` | Worker threads. More threads make the failure more likely |
| `--imports N` | `-i` | `2000` | Imports per thread |
| `--size-kib N` | `-s` | `3072` | Buffer size in KiB, a multiple of 256 |
| `--make-resident` | `-r` | off | Application-level workaround, see below |
| `--imports-only` | `-o` | off | Benchmark mode: import and free only, no kernel, see below |
| `--help` | `-h` | | Print usage |

Example: `./repro --threads 4 --imports 4000 --render-node /dev/dri/renderD129`

`--make-resident` binds each import right after it is created with `zeContextMakeMemoryResident`
(what torchcodec-xpu does). That path looks the allocation up by pointer, not by allocation id, so a
skipped scan no longer matters. It hides the race; it does not fix the driver.

`--imports-only` skips the kernel launch and the data check. Without a submit there is no residency
scan, so the race cannot occur and every driver passes. The `PASS` line then gives import/free
throughput, for comparing drivers.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | pass |
| 1 | Level Zero error, normally device lost |
| 2 | setup error or bad usage |
| 3 | kernel did not finish in 20 s (hang) |
| 4 | kernel read wrong data |

## Results (Arc Pro B70, `xe`, kernel 7.0)

Default options, 5 runs per driver. "Completed imports" is the total over all threads when the run ended.

| Driver | `--make-resident`<br>workaround | Threads | Imports per thread | Runs | Result | Error | Completed imports |
|---|---|---|---|---|---|---|---|
| release package 26.40.40196.8 | N | 8 | 2000 | 10 | 10 fail | `ZE_RESULT_ERROR_DEVICE_LOST` (`0x70000001`) | 10 to 2512 |
| system package, `libze_intel_gpu.so.1.17.39758` | N | 8 | 2000 | 5 | 5 fail | `ZE_RESULT_ERROR_DEVICE_LOST` (`0x70000001`) | 27 to 1346 |
| local build of master `5186791f8e` | N | 8 | 2000 | 5 | 5 fail | `ZE_RESULT_ERROR_DEVICE_LOST` (`0x70000001`) | 41 to 488 |
| local build of master `38c85b7889` | N | 8 | 2000 | 5 | 5 fail | `ZE_RESULT_ERROR_DEVICE_LOST` (`0x70000001`) | 148 to 5418 |
| system package, `libze_intel_gpu.so.1.17.39758` | Y | 8 | 2000 | 5 | 5 pass | N/A | 16000 (all) |
| local build of master `5186791f8e` | Y | 8 | 2000 | 5 | 5 pass | N/A | 16000 (all) |

On failure the program prints one extra line comparing the `xe` page faults logged since it started
(read from `/dev/kmsg`, so it needs root; in a container also pass `--device /dev/kmsg`) with
the imports in flight:

```
FAIL after 653 imports: kernel launch/sync -> 0x70000001
page faults: 101 distinct addresses, 101 inside imports in flight, 0 outside
```

In 10 of 10 failing runs (5 on the system package, 5 on master, 99 to 104 distinct faulted addresses
each), every faulted address lay inside the address range of an import that was in flight, none outside. The
faults are `FaultType: 0` (page not present) followed by `Fault response: Unsuccessful`. This shows
the faulting pages belong to imported allocations that have no bind; it does not by itself show why
the bind is missing.

## Notes

- The failure is probabilistic: it needs a submit on another thread to scan inside the window between
  the id bump and the insert, so more threads make it more likely. With 1 or 2 threads it did not
  show in 3 runs each.
- The same pattern (bump, then insert) exists in `importFdHandles` and other `allocationsCounter`
  sites.
- Written for `xe` only (it creates the dma-buf with xe ioctls).
