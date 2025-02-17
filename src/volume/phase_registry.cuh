/**
 * @file phase_registry.cuh
 * @author Qianyue He
 * @brief Includes all the derived class of PhaseFunction
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "volume/henyey_greenstein.cuh"
#include "volume/rayleigh.cuh"
#include "volume/sggx.cuh"
#include "core/enums.cuh"

template <typename PhaseType, typename... Args>
CPT_KERNEL void create_device_phase(PhaseFunction** dst, int index, Args... args) {
    static_assert(std::is_base_of_v<PhaseFunction, PhaseType>, 
                  "PhaseType must be derived from PhaseFunction");

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (dst[index]) delete dst[index];
        dst[index] = new PhaseType(args...);
    }
}

CPT_KERNEL void load_phase_kernel(PhaseFunction** dst, int index, Vec4 data);