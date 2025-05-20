// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * Utilization function to copy the classes with virtual functions to GPU
 * @author: Qianyue He
 * @date:   2024.5.22
 */

#pragma once
#include "bsdf/bsdf.cuh"
#include "core/emitter.cuh"

template <typename BSDFType>
CPT_KERNEL void create_bsdf(BSDF **dst, Vec4 k_d, Vec4 k_s, Vec4 k_g,
                            int flags = ScatterStateFlag::BSDF_DIFFUSE) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (*dst)
            delete *dst;
        *dst = new BSDFType();
        (*dst)->set_kd(std::move(k_d));
        (*dst)->set_ks(std::move(k_s));
        (*dst)->set_kg(std::move(k_g));
        (*dst)->set_lobe(flags);
    }
}

// This kernel function is to set the general BSDF Params, when vptr and vtable
// are built
template <typename BSDFType>
CPT_KERNEL void load_bsdf(BSDF **dst, Vec4 k_d, Vec4 k_s, Vec4 k_g) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*dst)->set_kd(std::move(k_d));
        (*dst)->set_ks(std::move(k_s));
        (*dst)->set_kg(std::move(k_g));
    }
}

CPT_KERNEL void create_metal_bsdf(BSDF **dst, Vec3 eta_t, Vec3 k, Vec4 k_g,
                                  float roughness_x, float roughness_y);

CPT_KERNEL void load_metal_bsdf(BSDF **dst, Vec3 eta_t, Vec3 k, Vec4 k_g,
                                float roughness_x, float roughness_y);

CPT_KERNEL void load_dispersion_bsdf(BSDF **dst, Vec4 k_s, float index_a,
                                     float index_b);

CPT_KERNEL void create_dispersion_bsdf(BSDF **dst, Vec4 k_s, float index_a,
                                       float index_b);

CPT_KERNEL void create_forward_bsdf(BSDF **dst);

template <typename PlasticType>
CPT_KERNEL void
create_plastic_bsdf(BSDF **dst, Vec4 k_d, Vec4 k_s, Vec4 sigma_a, float ior,
                    float trans_scaler = 1.f, float thickness = 0,
                    bool penetrable = false) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (*dst)
            delete *dst;
        *dst = new PlasticType(k_d, k_s, sigma_a, ior, trans_scaler, thickness,
                               penetrable);
        (*dst)->set_kd(std::move(k_d));
        (*dst)->set_ks(std::move(k_s));
        (*dst)->set_kg(std::move(sigma_a));
    }
}

template <typename PlasticType>
CPT_KERNEL void load_plastic_bsdf(BSDF **dst, Vec4 k_d, Vec4 k_s, Vec4 sigma_a,
                                  float ior, float trans_scaler = 1.f,
                                  float thickness = 0,
                                  bool penetrable = false) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // I will make sure (I can) the base ptr is actually of PlasticType*
        // So dynamic_cast is not needed (actually, not allowed on device code)
        PlasticType *ptr = static_cast<PlasticType *>(*dst);
        ptr->eta = 1.f / ior;
        ptr->trans_scaler = trans_scaler;
        ptr->thickness = thickness;
        ptr->__padding = penetrable;
        ptr->set_kd(std::move(k_d));
        ptr->set_ks(std::move(k_s));
        ptr->set_kg(std::move(sigma_a));
    }
}

template <typename Ty> CPT_KERNEL void destroy_gpu_alloc(Ty **dst) {
    delete dst[threadIdx.x];
}

CPT_KERNEL void create_point_source(Emitter *&dst, Vec4 le, Vec3 pos);
CPT_KERNEL void create_area_source(Emitter *&dst, Vec4 le, int obj_ref,
                                   bool is_sphere,
                                   cudaTextureObject_t obj = NULL);
CPT_KERNEL void create_area_spot_source(Emitter *&dst, Vec4 le, float cos_val,
                                        int obj_ref, bool is_sphere,
                                        cudaTextureObject_t obj = NULL);
CPT_KERNEL void create_envmap_source(Emitter *&dst, cudaTextureObject_t obj,
                                     float scaler = 1, float azimuth = 0,
                                     float zenith = 0);
CPT_KERNEL void create_abstract_source(Emitter *&dst);
CPT_KERNEL void set_emission(Emitter *&dst, Vec3 color, float scaler = 1.f);

CPT_KERNEL void call_setter(Emitter *&dst, float v1, float v2, float v3);
