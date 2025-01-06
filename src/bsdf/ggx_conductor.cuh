/**
 * @file ggx_conductor.cuh
 * @author Qianyue He
 * @brief GGX microfacet normal distribution based BSDF
 * @date 2025-01-06
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "bsdf/bsdf.cuh"
#include "bsdf/fresnel.cuh"
#include "core/textures.cuh"

class GGXConductorBSDF: public BSDF {
/**
 * @brief GGX microfacet normal distribution based BSDF
 * k_d is the eta_t of the metal
 * k_s is the k (Vec3) and the mapped roughness (k_s[3])
 * k_g is the underlying color (albedo)
 */
public:
    using BSDF::k_s;
    FresnelTerms fresnel;
public:
    CPT_CPU_GPU GGXConductorBSDF(Vec3 eta_t, Vec3 k, Vec4 albedo, float roughness_x, float roughness_y);

    CPT_CPU_GPU GGXConductorBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */, int index) const override;

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
    ) const override;
};