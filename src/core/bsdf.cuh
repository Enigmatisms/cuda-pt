/**
 * CUDA object information
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/vec3.cuh"

class BSDF {
private:
    Vec3 k_d;
    Vec3 k_s;
    Vec3 k_g;
public:
    CPT_CPU_GPU BSDF() {}
};