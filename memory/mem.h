//
// Created by Mrinal Chaturvedi on 10/02/26.
//

#ifndef FLUXIONS_MEM_H
#define FLUXIONS_MEM_H
#include <vector>
namespace fluxions {
    struct memAllocator {
        const int align;
        explicit memAllocator(int align): align(align) {}
        memAllocator(const memAllocator&)=delete;
        memAllocator& operator=(const memAllocator&)=delete;
        virtual void* malloc(std::size_t n)=0;
        virtual void free(void* mem)=0;
        virtual void zero(void *p, size_t n)=0;
        std::size_t round_up_align(std::size_t n) const {
            if (align < 2) return n;
            return ((n + align - 1) / align) * align;
        }
        virtual ~memAllocator();
    };

    struct CPUAllocator: public memAllocator{
        CPUAllocator(): memAllocator(32){}
        void *malloc(std::size_t n) override;
        void free(void *mem) override;
        void zero(void *p,size_t n) override;
    };
}
#endif //FLUXIONS_MEM_H