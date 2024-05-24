/**
 * Random value generator CUDA
 * @date: 5.5.2024
 * @author: Qianyue He
 * 
 * Note that CUDA supports Sobol sequence
 * therefore... well, very interesting
*/
#pragma once
#include <curand_kernel.h>
#include "core/cuda_utils.cuh"
#include "core/vec2.cuh"

class Sampler {
public:
    CPT_GPU Sampler(int seed, int offset = 0) {
        curand_init(seed + offset, 0, 0, &rand_state);
    }

    CPT_GPU Vec2 next2D() noexcept { return Vec2(curand_uniform(&rand_state), curand_uniform(&rand_state)); }
    CPT_GPU float next1D() noexcept { return curand_uniform(&rand_state); }
    CPT_GPU float next1D_n() noexcept { return curand_normal(&rand_state); }
    CPT_GPU uint32_t discrete1D() noexcept { return curand(&rand_state); }
private:
    curandState rand_state;
};