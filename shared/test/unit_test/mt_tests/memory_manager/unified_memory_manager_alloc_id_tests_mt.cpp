/*
 * Copyright (C) 2026 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "shared/source/memory_manager/unified_memory_manager.h"
#include "shared/source/memory_manager/unified_memory_properties.h"
#include "shared/test/common/mocks/mock_device.h"
#include "shared/test/common/mocks/mock_graphics_allocation.h"
#include "shared/test/common/mocks/mock_memory_manager.h"
#include "shared/test/common/mocks/mock_svm_manager.h"
#include "shared/test/common/mocks/ult_device_factory.h"

#include "gtest/gtest.h"

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

using namespace NEO;

// An allocation id must never be visible (counter advanced) before the allocation is published in
// internalAllocationsMap: makeIndirectAllocationsResident() advances its high-water mark to the counter
// and never revisits an id, so an id that was visible too early is never made resident.
struct SvmAllocIdPublicationMtTest : ::testing::Test {
    void SetUp() override {
        deviceFactory = std::make_unique<UltDeviceFactory>(1, 1);
        device = deviceFactory->rootDevices[0];
        svmManager = std::make_unique<MockSVMAllocsManager>(device->getMemoryManager());
    }

    // Runs allocate() on another thread while this thread holds the container lock and checks that the
    // id was neither taken nor published until the lock was released, and that both happened together.
    void *allocateWhileContainerLockedAndVerify(const std::function<void *()> &allocate) {
        const auto counterBefore = svmManager->allocationsCounter.load();
        const auto mapSizeBefore = svmManager->internalAllocationsMap.size();

        std::atomic<bool> started = false;
        std::atomic<bool> finished = false;
        void *ptr = nullptr;
        std::optional<SVMAllocsManager::ContainerReadLockTypeRAIIHelper> containerLock;
        containerLock.emplace(*svmManager);

        std::thread allocator([&] {
            started = true;
            ptr = allocate();
            finished = true;
        });
        while (!started) {
            std::this_thread::yield();
        }
        for (int i = 0; i < spinsWhileBlocked && !finished; i++) {
            std::this_thread::yield();
        }
        EXPECT_FALSE(finished);
        EXPECT_EQ(counterBefore, svmManager->allocationsCounter.load());
        EXPECT_EQ(mapSizeBefore, svmManager->internalAllocationsMap.size());

        containerLock.reset();
        allocator.join();

        EXPECT_NE(nullptr, ptr);
        EXPECT_EQ(counterBefore + 1, svmManager->allocationsCounter.load());
        auto allocData = svmManager->getSVMAlloc(ptr);
        EXPECT_NE(nullptr, allocData);
        if (allocData) {
            EXPECT_EQ(counterBefore + 1, allocData->getAllocId());
            EXPECT_EQ(1u, svmManager->internalAllocationsMap.count(allocData->getAllocId()));
        }
        return ptr;
    }

    static constexpr int spinsWhileBlocked = 20000;
    std::unique_ptr<UltDeviceFactory> deviceFactory;
    Device *device = nullptr;
    std::unique_ptr<MockSVMAllocsManager> svmManager;
    RootDeviceIndicesContainer rootDeviceIndices{0u};
    std::map<uint32_t, DeviceBitfield> deviceBitfields{{0u, DeviceBitfield(1)}};
};

TEST_F(SvmAllocIdPublicationMtTest, givenDeviceAllocationWhenContainerLockedThenIdTakenAndPublishedTogether) {
    UnifiedMemoryProperties properties(InternalMemoryType::deviceUnifiedMemory, 1, rootDeviceIndices, deviceBitfields);
    properties.device = device;

    auto ptr = allocateWhileContainerLockedAndVerify([&] { return svmManager->createUnifiedMemoryAllocation(4096, properties); });
    svmManager->freeSVMAlloc(ptr);
}

TEST_F(SvmAllocIdPublicationMtTest, givenHostAllocationWhenContainerLockedThenIdTakenAndPublishedTogether) {
    UnifiedMemoryProperties properties(InternalMemoryType::hostUnifiedMemory, 1, rootDeviceIndices, deviceBitfields);

    auto ptr = allocateWhileContainerLockedAndVerify([&] { return svmManager->createHostUnifiedMemoryAllocation(4096, properties); });
    svmManager->freeSVMAlloc(ptr);
}

TEST_F(SvmAllocIdPublicationMtTest, givenImportedFdHandleWhenContainerLockedThenIdTakenAndPublishedTogether) {
    auto memoryManager = static_cast<MockMemoryManager *>(device->getMemoryManager());
    auto ptr = allocateWhileContainerLockedAndVerify([&] {
        SvmAllocationData mappedPeerAllocData(device->getRootDeviceIndex());
        return memoryManager->importFdHandle(device, svmManager.get(), 1u, AllocationType::buffer, false, nullptr, nullptr, mappedPeerAllocData, false, false, 0u);
    });
    svmManager->freeSVMAlloc(ptr);
}

TEST_F(SvmAllocIdPublicationMtTest, givenAllocationDataWhenAssigningIdAndInsertingThenIdTakenAndPublishedTogether) {
    MockGraphicsAllocation graphicsAllocation(0u, 1u /*num gmms*/, AllocationType::buffer, reinterpret_cast<void *>(0x1000), 0x1000, 0, 4096u, MemoryPool::system64KBPages, MemoryManager::maxOsContextCount);
    SvmAllocationData allocData(device->getRootDeviceIndex());
    allocData.gpuAllocations.addAllocation(&graphicsAllocation);
    allocData.size = 4096u;

    allocateWhileContainerLockedAndVerify([&] {
        svmManager->assignIdAndInsertSVMAlloc(allocData);
        return reinterpret_cast<void *>(graphicsAllocation.getGpuAddress());
    });
    svmManager->removeSVMAlloc(allocData);
}

// Stress: with the container lock held, every id up to the counter must already be published, because
// that is exactly what makeIndirectAllocationsResident() sees when it scans.
TEST_F(SvmAllocIdPublicationMtTest, givenConcurrentAllocationsWhenObservingUnderContainerLockThenEveryIdUpToCounterIsPublished) {
    UnifiedMemoryProperties properties(InternalMemoryType::deviceUnifiedMemory, 1, rootDeviceIndices, deviceBitfields);
    properties.device = device;

    constexpr size_t allocationsPerThread = 100;
    constexpr size_t allocatorThreads = 4;
    std::atomic<bool> allocatorsDone = false;
    std::atomic<size_t> unpublishedIdsSeen = 0;

    std::vector<std::vector<void *>> allocated(allocatorThreads);
    std::vector<std::thread> allocators;
    for (size_t t = 0; t < allocatorThreads; t++) {
        allocators.emplace_back([&, t] {
            for (size_t i = 0; i < allocationsPerThread; i++) {
                allocated[t].push_back(svmManager->createUnifiedMemoryAllocation(4096, properties));
            }
        });
    }
    std::thread observer([&] {
        while (!allocatorsDone) {
            auto lock = svmManager->obtainReadContainerLock();
            for (uint32_t id = 1; id <= svmManager->allocationsCounter.load(); id++) {
                if (svmManager->internalAllocationsMap.count(id) == 0) {
                    unpublishedIdsSeen++;
                }
            }
        }
    });
    for (auto &allocator : allocators) {
        allocator.join();
    }
    allocatorsDone = true;
    observer.join();

    EXPECT_EQ(0u, unpublishedIdsSeen.load());
    EXPECT_EQ(allocatorThreads * allocationsPerThread, svmManager->allocationsCounter.load());
    for (auto &perThread : allocated) {
        for (auto ptr : perThread) {
            svmManager->freeSVMAlloc(ptr);
        }
    }
}
