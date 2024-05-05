/**
 * Random value generator CUDA
 * @date: 5.5.2024
 * @author: Qianyue He
 * 
 * Note that CUDA supports Sobol sequence
 * therefore... well, very interesting
*/
#include <curand_kernel.h>
#include "core/cuda_utils.cuh"

class Sampler {
public:
    CPT_GPU Sampler(int seed) {
        curand_init(seed, 0, 0, &rand_state);

    }

    CPT_GPU float rand() { return curand_uniform(&rand_state); }
    CPT_GPU float randn() { return curand_normal(&rand_state); }
    CPT_GPU int randint() { return curand(&rand_state); }
private:
    curandState rand_state;
};