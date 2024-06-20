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

class TinySampler {
struct RandState {
    unsigned int v[4], d[2];
};
public:
    CPT_CPU_GPU TinySampler(int seed, int offset = 0) {
        _init_state(seed + offset);
    }

    CPT_CPU_GPU_INLINE Vec2 next2D() noexcept { return Vec2(
            _uniform_uint_to_float(discrete1D()),
            _uniform_uint_to_float(discrete1D())
        ); 
    }
    CPT_CPU_GPU_INLINE float next1D() noexcept { return _uniform_uint_to_float(discrete1D()); }

    CPT_CPU_GPU_INLINE uint32_t discrete1D()
    {
        unsigned int t = (rand_state.v[0] ^ (rand_state.v[0] >> 2));
        // TODO: optimize this?
        rand_state.v[0] = rand_state.v[1];
        rand_state.v[1] = rand_state.v[2];
        rand_state.v[2] = rand_state.v[3];
        rand_state.v[3] = rand_state.d[0];
        rand_state.d[0] = (rand_state.d[0] ^ (rand_state.d[0] <<4)) ^ (t ^ (t << 1));
        rand_state.d[1] += 362437;
        return rand_state.d[0] + rand_state.d[1];
    }
private:
    static CPT_CPU_GPU_INLINE float _uniform_uint_to_float(unsigned int x)
    {
        return x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    }

    CPT_CPU_GPU_INLINE void _init_state(unsigned long long seed) {
        // Break up seed, apply salt
        // Constants are arbitrary nonzero values
        unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
        unsigned int s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;
        // Simple multiplication to mix up bits
        // Constants are arbitrary odd values
        unsigned int t0 = 1099087573UL * s0;
        unsigned int t1 = 2591861531UL * s1;
        rand_state.v[0] = 123456789UL + t0;
        rand_state.v[1] = 362436069UL ^ t0;
        rand_state.v[2] = 521288629UL + t1;
        rand_state.v[3] = 88675123UL ^ t1;
        rand_state.d[0] = 5783321UL + t0;
        rand_state.d[1] = 6615241 + t1 + t0;
    }


    
private:
    RandState rand_state;
};

class CudaSampler {
public:
    CPT_GPU CudaSampler(int seed, int offset = 0) {
        curand_init(seed + offset, 0, 0, &rand_state);
        sizeof(rand_state);
    }

    CPT_GPU Vec2 next2D() noexcept { return Vec2(curand_uniform(&rand_state), curand_uniform(&rand_state)); }
    CPT_GPU float next1D() noexcept { return curand_uniform(&rand_state); }
    CPT_GPU uint32_t discrete1D() noexcept { return curand(&rand_state); }
private:
    curandState rand_state;
};