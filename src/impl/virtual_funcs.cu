/**
 * Utilization function to copy the classes with virtual functions to GPU
 * Implementation
 * @author: Qianyue He
 * @date:   2024.9.6
*/

#pragma once
#include "core/virtual_funcs.cuh"

CPT_KERNEL void create_point_source(Emitter* &dst, Vec4 le, Vec3 pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst = new PointSource(std::move(le), std::move(pos));
    }
}

CPT_KERNEL void create_area_source(Emitter* &dst, Vec4 le, int obj_ref, bool is_sphere) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst = new AreaSource(std::move(le), obj_ref, is_sphere);
    }
}

CPT_KERNEL void create_abstract_source(Emitter* &dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst = new Emitter(Vec4(0, 0, 0));
    }
}
