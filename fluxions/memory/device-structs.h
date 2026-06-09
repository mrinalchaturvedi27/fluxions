//
// Created by Mrinal Chaturvedi on 15/02/26.
//

#ifndef FLUXIONS_DEVICE_STRUCTS_H
#define FLUXIONS_DEVICE_STRUCTS_H

namespace fluxions {
    enum class DeviceType {CPU, GPU};
/*
 * FXS   -> forward pass memory
 * DEDFS -> backward pass memory
 * PS    -> parameter memory
 * SCS   -> scratch memory (for use in temporary calculations)
 * NONE  -> when a memory pool has not been assigned yet
 */
    enum class DeviceMempool {FXS = 0, DEDFS = 1, PS = 2, SCS = 3, NONE = 4};
    struct ComputationGraph; // to resolve cyclic dependency
    struct Tensor;

    struct DeviceMempoolSizes {
        std::size_t used[4];
        DeviceMempoolSizes() = default;
        DeviceMempoolSizes(std::size_t total_s);
        DeviceMempoolSizes(std::size_t fxs_s, std::size_t dEdfs_s, std::size_t ps_s, std::size_t sc_s);
        DeviceMempoolSizes(const std::string & descriptor);
    };

}
#endif //FLUXIONS_DEVICE_STRUCTS_H