#include "volume/homogeneous.cuh"

CPT_KERNEL void create_homogeneous_volume(
    Medium** media,
    PhaseFunction** phases,
    int med_id, int ph_id,
    Vec4 sigma_a, 
    Vec4 sigma_s,
    float scale
) {
    if (threadIdx.x == 0) {
        media[med_id] = new HomogeneousMedium(sigma_a * scale, sigma_s * scale);
        media[med_id]->bind_phase_function(phases[ph_id]);
    }
}