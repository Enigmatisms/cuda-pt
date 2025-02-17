#include "volume/phase_registry.cuh"

CPT_KERNEL void load_phase_kernel(PhaseFunction** dst, int index, Vec4 data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[index]->set_param(std::move(data));
    }
}