/**
 * Utilization function to copy the classes with virtual functions to GPU
 * @author: Qianyue He
 * @date:   2024.5.22
*/

#pragma once
#include "core/bsdf.cuh"
#include "core/emitter.cuh"

template <typename BSDFType>
__global__ void create_bsdf(BSDF* &dst, Vec3 k_d, Vec3 k_s, Vec3 k_g, int kd_tex_id = 0, int ex_tex_id = 0) {
    dst = new BSDFType(k_d, k_s, k_g, kd_tex_id, ex_tex_id);
}

__global__ void create_point_source(Emitter* &dst, Vec3 le, Vec3 pos) {
    dst = new PointSource(std::move(le), std::move(pos));
}

__global__ void create_area_source(Emitter* &dst, Vec3 le, int obj_ref) {
    dst = new AreaSource(std::move(le), obj_ref);
}

__global__ void create_abstract_source(Emitter* &dst) {
    dst = new Emitter(Vec3(0, 0, 0), -1);
}

template <typename Ty>
__global__ void destroy_gpu_alloc(Ty** dst) {
    delete dst[threadIdx.x];
}
