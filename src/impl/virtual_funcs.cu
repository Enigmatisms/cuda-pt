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

CPT_KERNEL void create_area_source(Emitter* &dst, Vec4 le, int obj_ref, bool is_sphere, cudaTextureObject_t obj) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst = new AreaSource(std::move(le), obj_ref, is_sphere);
    }
}

CPT_KERNEL void create_envmap_source(Emitter* &dst, cudaTextureObject_t obj, float scaler, float azimuth, float zenith) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst = new EnvMapEmitter(obj, scaler, azimuth, zenith);
    }
}

CPT_KERNEL void set_emission(Emitter* &dst, Vec3 color, float scaler) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst->set_le(std::move(color), scaler);
    }
}

CPT_KERNEL void call_setter(Emitter* &dst, float v1, float v2, float v3) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst->set_func1(v1);
        dst->set_func2(v2);
        dst->set_func3(v3);
    }
}

CPT_KERNEL void create_abstract_source(Emitter* &dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst = new Emitter(Vec4(0, 0, 0));
    }
}

CPT_KERNEL void create_metal_bsdf(BSDF** dst, Vec3 eta_t, Vec3 k, Vec4 k_g, float roughness_x, float roughness_y) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dst = new GGXConductorBSDF(eta_t, k, k_g, roughness_x, roughness_y);
    }
}

CPT_KERNEL void load_metal_bsdf(
    BSDF** dst, Vec3 eta_t, Vec3 k, Vec4 k_g, float roughness_x, float roughness_y
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // I will make sure (I can) the base ptr is actually of PlasticType*
        // So dynamic_cast is not needed (actually, not allowed on device code)
        GGXConductorBSDF* ptr = static_cast<GGXConductorBSDF*>(*dst);
        ptr->fresnel = FresnelTerms(std::move(eta_t), std::move(k));
        ptr->set_kd(Vec4(0));
        ptr->set_ks(Vec4(roughness_to_alpha(roughness_x), roughness_to_alpha(roughness_y), 1));
        ptr->set_kg(std::move(k_g));
    }
}

CPT_KERNEL void load_dispersion_bsdf(
    BSDF** dst, Vec4 k_s, float index_a, float index_b
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*dst)->set_kd(Vec4(index_a, index_b, 0));
        (*dst)->set_ks(std::move(k_s));
    }
}

CPT_KERNEL void create_dispersion_bsdf(
    BSDF** dst, Vec4 k_s, float index_a, float index_b
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *dst = new DispersionBSDF(k_s, index_a, index_b);
        (*dst)->set_kd(Vec4(index_a, index_b, 0));
        (*dst)->set_ks(std::move(k_s));
    }
}
