//
// Created by Mrinal Chaturvedi on 14/02/26.
//
#include "mem.h"
#ifndef FLUXIONS_MEM_POOL_H
#define FLUXIONS_MEM_POOL_H
namespace fluxions {
    class InternalMemoryPool {
    private:
        std::string name;
        size_t capacity=0;
        memAllocator* a;
        void* mem=nullptr;
        size_t used;


        void sys_alloc(size_t cap);

        void zero_all() {
        a->zero(mem,capacity);
        }

    public:
        explicit InternalMemoryPool(const std::string &name,size_t cap,memAllocator *a): name(name),a(a) {
            sys_alloc(cap);
            zero_all();
        }
        ~InternalMemoryPool() {
            if (mem) {
                a->free(mem);
            }
        }

        void* allocate(size_t n);

        void free() {
            used=0;
        }

        void zero_allocate_memory() {
            if (used==0) return;
            a->zero(mem,used);
        }
    };

    class AlignedMemoryPool {
    private:
        std::string name;
        size_t cap;
        int current;
        memAllocator* a;
        InternalMemoryPool* pool;

    public:
        explicit AlignedMemoryPool(std::string &name,size_t cap,memAllocator *a);

        ~AlignedMemoryPool();

        void* allocate(size_t n);

        void free();

        void zero_allocated_memory();

        size_t used();

        void set_used(size_t s);

        size_t get_cap();
    };
}
#endif //FLUXIONS_MEM_POOL_H