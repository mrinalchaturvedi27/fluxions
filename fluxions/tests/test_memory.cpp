//
// Created by Mrinal Chaturvedi on 14/02/26.
//
#include <gtest/gtest.h>
#include "mem_pool.h"
#include "mem.h"

using namespace fluxions;

TEST(InternalMemoryPoolTest, AllocationReturnsPointer) {
    CPUAllocator alloc;
    InternalMemoryPool pool("test", 1024, &alloc);

    void* p = pool.allocate(100);

    EXPECT_NE(p, nullptr);
}

TEST(InternalMemoryPoolTest, AllocationMovesForward) {
    CPUAllocator alloc;
    InternalMemoryPool pool("test", 1024, &alloc);

    void* p1 = pool.allocate(100);
    void* p2 = pool.allocate(100);

    EXPECT_GT(p2, p1);
}

TEST(InternalMemoryPoolTest, ResetReusesMemory) {
    CPUAllocator alloc;
    InternalMemoryPool pool("test", 1024, &alloc);

    void* p1 = pool.allocate(100);
    pool.free();
    void* p2 = pool.allocate(100);

    EXPECT_EQ(p1, p2);
}

TEST(InternalMemoryPoolTest, OverflowFails) {
    CPUAllocator alloc;
    InternalMemoryPool pool("test", 128, &alloc);

    void* p1 = pool.allocate(200);

    EXPECT_EQ(p1, nullptr);
}

TEST(InternalMemoryPoolTest, AlignmentCorrect) {
    CPUAllocator alloc;
    InternalMemoryPool pool("test", 1024, &alloc);

    void* p = pool.allocate(13);

    uintptr_t addr = reinterpret_cast<uintptr_t>(p);

    EXPECT_EQ(addr % alloc.align, 0);
}
