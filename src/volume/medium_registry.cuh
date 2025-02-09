/**
 * @file medium_registry.cuh
 * @author Qianyue He
 * @brief Includes all the derived class of Medium
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "volume/homogeneous.cuh"
#include "volume/grid.cuh"
#include "core/enums.cuh"

template <typename MedType, typename... Args>
CPT_KERNEL void create_device_medium(Medium** dst, Args... args) {
    static_assert(std::is_base_of_v<Medium, MedType>, 
                  "MedType must be derived from Medium");

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dst = new MedType(args...);
    }
}
