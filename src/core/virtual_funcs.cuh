/**
 * Utilization function to copy the classes with virtual functions to GPU
 * @author: Qianyue He
 * @date:   2024.5.22
*/

#pragma once
#include "core/bsdf.cuh"
#include "core/emitter.cuh"

template <typename BSDFType>
CPT_KERNEL void create_bsdf(BSDF** dst, Vec4 k_d, Vec4 k_s, Vec4 k_g, int flags = BSDFFlag::BSDF_DIFFUSE) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dst = new BSDFType();
        (*dst)->set_kd(std::move(k_d));
        (*dst)->set_ks(std::move(k_s));
        (*dst)->set_kg(std::move(k_g));
        (*dst)->set_lobe(flags);
    }
}

// This kernel function is to set the general BSDF Params, when vptr and vtable are built
template <typename BSDFType>
CPT_KERNEL void load_bsdf(BSDF** dst, Vec4 k_d, Vec4 k_s, Vec4 k_g) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*dst)->set_kd(std::move(k_d));
        (*dst)->set_ks(std::move(k_s));
        (*dst)->set_kg(std::move(k_g));
    }
}

CPT_KERNEL void create_metal_bsdf(
    BSDF** dst, Vec3 eta_t, Vec3 k, Vec4 k_g, float roughness_x, float roughness_y
);

CPT_KERNEL void load_metal_bsdf(
    BSDF** dst, Vec3 eta_t, Vec3 k, Vec4 k_g, float roughness_x, float roughness_y
);

template <typename PlasticType>
CPT_KERNEL void create_plastic_bsdf(
    BSDF** dst, Vec4 k_d, Vec4 k_s, Vec4 sigma_a, 
    float ior, float trans_scaler = 1.f, 
    float thickness = 0
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dst = new PlasticType(k_d, k_s, sigma_a, ior, trans_scaler, thickness);
        (*dst)->set_kd(std::move(k_d));
        (*dst)->set_ks(std::move(k_s));
        (*dst)->set_kg(std::move(sigma_a));
    }
}

template <typename PlasticType>
CPT_KERNEL void load_plastic_bsdf(
    BSDF** dst, Vec4 k_d, Vec4 k_s, Vec4 sigma_a, 
    float ior, float trans_scaler = 1.f, 
    float thickness = 0
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // I will make sure (I can) the base ptr is actually of PlasticType*
        // So dynamic_cast is not needed (actually, not allowed on device code)
        PlasticType* ptr = static_cast<PlasticType*>(*dst);
        ptr->eta = 1.f / ior;
        ptr->trans_scaler = trans_scaler;
        ptr->thickness = thickness;
        ptr->set_kd(std::move(k_d));
        ptr->set_ks(std::move(k_s));
        ptr->set_kg(std::move(sigma_a));
    }
}

template <typename Ty>
CPT_KERNEL void destroy_gpu_alloc(Ty** dst) {
    delete dst[threadIdx.x];
}

CPT_KERNEL void create_point_source(Emitter* &dst, Vec4 le, Vec3 pos);
CPT_KERNEL void create_area_source(Emitter* &dst, Vec4 le, int obj_ref, bool is_sphere, cudaTextureObject_t obj = NULL);
CPT_KERNEL void create_envmap_source(Emitter* &dst, cudaTextureObject_t obj, float scaler = 1, float azimuth = 0, float zenith = 0);
CPT_KERNEL void create_abstract_source(Emitter* &dst);
CPT_KERNEL void set_emission(Emitter* &dst, Vec3 color, float scaler = 1.f);

CPT_KERNEL void call_setter(Emitter* &dst, float v1, float v2, float v3);