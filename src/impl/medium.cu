/**
 * @file medium.cu
 * @author Qianyue He
 * @date 2025-2-17
 * @copyright Copyright (c) 2024
 */
#include "core/medium.cuh"

const std::array<const char*, NumSupportedMedium> MEDIUM_NAMES = {
    "Homogeneous",     
    "Grid Volume" 
};

const std::array<const char*, NumSupportedPhase> PHASES_NAMES = {
    "NullForward",     
    "Isotropic",
    "HenyeyGreenstein",
    "DuoHG",
    "Rayleigh",
    "SGGX"
};