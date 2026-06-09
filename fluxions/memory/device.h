//
// Created by Mrinal Chaturvedi on 15/02/26.
//
#pragma once
#include "mem.h"
#include "mem_pool.h"
#include "device-structs.h"
#include <vector>

#ifndef FLUXIONS_REPO_DEVICE_H
#define FLUXIONS_REPO_DEVICE_H
namespace fluxions {
    class Device {
    protected:
        Device(int i,DeviceType t,memAllocator *): device_id(i),type(t),mem(m),pools(4,nullptr){}
        Device(const Device &)=delete;
        virtual ~Device();

    public:
        memAllocator *mem;
        std::vector<AlignedMemoryPool*> pools;
        int device_id;
        DeviceType type;
        float* kSCALAR_MINUSONE;
        float* kSCALAR_ONE;
        float* kSCALAR_ZERO;
        std::string name;
        void allocate_tensor(DeviceMempool mem_pool,Tensor &tensor);
        virtual DeviceMempoolSizes mark(ComputationGraph *cg);
        virtual void revert(const DeviceMempoolSizes & cp);
    };
}

#endif //FLUXIONS_REPO_DEVICE_H