#include "mem.h"
#include <iostream>
#include <mm_malloc.h>
using namespace std;

namespace fluxions {
    memAllocator::~memAllocator(){}

    void* CPUAllocator::malloc(std::size_t n) {
        void *ptr=_mm_malloc(n,align);
        if (!ptr) {
          cerr<<"CPU memory allocation failed n=" << n << " align=" << align << endl;
        }
        return ptr;
    }
    void CPUAllocator::free(void* mem) {
        _mm_free(mem);
    }

    void CPUAllocator::zero(void* p, size_t n) {
        memset(p, 0, n);
    }
}
