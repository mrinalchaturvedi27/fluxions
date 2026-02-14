#include "mem_pool.h"
#include <iostream>
using namespace fluxions;
using namespace std;

void* InternalMemoryPool::allocate(size_t n) {
    auto rounded_n = a->round_up_align(n);
    if (rounded_n + used > capacity) {
        return 0;
    }
    void* res = static_cast<char*>(mem) + used;
    used += rounded_n;
    return res;
}

void InternalMemoryPool::sys_alloc(size_t cap) {
    capacity = a->round_up_align(cap);
    mem = a->malloc(capacity);
    if (mem == NULL)
      cerr<< name << " failed to allocate " << capacity;
    used = 0;
}

AlignedMemoryPool::AlignedMemoryPool(std::string &name,size_t cap,memAllocator *a): name(name),cap(cap),a(a) {
    if (cap == 0) {
        std::cerr << "Cannot create memory pool with zero capacity\n";
        std::abort();
    }
    pool = new InternalMemoryPool(name, cap, a);
}

AlignedMemoryPool::~AlignedMemoryPool() {
    delete pool;
}

void* AlignedMemoryPool::allocate(size_t n) {
    void* res = pool->allocate(n);
    if (!res) {
        std::cerr << "Memory pool exhausted in " << name << std::endl;
        return nullptr;  // or throw std::bad_alloc()
    }
    return res;
}

void AlignedMemoryPool::free() {
 pool->free();
}


size_t AlignedMemoryPool::get_cap() {
    return cap;
}

