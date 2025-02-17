#include "volume/medium_registry.cuh"

CPT_KERNEL void load_homogeneous_kernel(Medium** dst, Vec4 sigma_a, Vec4 sigma_s, float scale) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        HomogeneousMedium* ptr = static_cast<HomogeneousMedium*>(*dst);
        ptr->set_params(sigma_a * scale, sigma_s * scale);
    }
}

CPT_KERNEL void load_grid_kernel(Medium** dst, Vec4 const_alb, float scale, float tp_scale, float em_scale) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        GridVolumeMedium* ptr = static_cast<GridVolumeMedium*>(*dst);
        ptr->set_params(std::move(const_alb), scale, tp_scale, em_scale);
    }
}

CPT_KERNEL void bind_phase_func_kernel(
    Medium** dst, 
    PhaseFunction** phases, 
    int ph_id
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*dst)->bind_phase_function(phases[ph_id]);
    }
}