/**
 * @file bsdf.cu
 * @author Qianyue He
 * @date 2024-11-06
 * @copyright Copyright (c) 2024
 */
#include "bsdf/bsdf.cuh"

const std::array<const char*, NumSupportedBSDF> BSDF_NAMES = {
    "Lambertian",     
    "Specular",       
    "Translucent",    
    "Plastic",        
    "PlasticForward", 
    "GGXConductor",
    "Dispersion",
    "Forward (Null)"
};