#include "volume/homogeneous.cuh"

CPT_KERNEL void create_homogeneous_volume(
    Medium** media,
    Vec4 sigma_a, 
    Vec4 sigma_s,
    float scale,
    PhaseFunction* ptr
) {
    if (threadIdx.x == 0) {
        *media = new HomogeneousMedium(sigma_a * scale, sigma_s * scale);
        (*media)->bind_phase_function(ptr);
    }
}