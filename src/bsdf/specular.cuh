/**
 * @file specular.cuh
 * @author Qianyue He
 * @brief Mirror Specular BSDF
 * @date 2025-01-06
 * @copyright Copyright (c) 2025
 */
#pragma once

#include "bsdf/bsdf.cuh"
#include "core/textures.cuh"

class SpecularBSDF: public BSDF {
public:
    using BSDF::k_s;
    CPT_CPU_GPU SpecularBSDF(Vec4 _k_s):
        BSDF(Vec4(0, 0, 0), std::move(_k_s), Vec4(0, 0, 0), BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_REFLECT) {}

    CPT_CPU_GPU SpecularBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */, int) const override {
        return 0.f;
    }

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        auto ref_dir = in.advance(normal, -2.f * in.dot(normal)).normalized();
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        return c_textures.eval(spec_tex, it.uv_coord, k_s) * (out.dot(ref_dir) > 0.99999f);
    }

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
    ) const override {
        // throughput *= f / pdf
        samp_lobe = static_cast<BSDFFlag>(bsdf_flag);
        const Vec3 normal = c_textures.eval_normal(it, index);
        float in_dot_n = indir.dot(normal);
        pdf = 1.f;
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        throughput *= c_textures.eval(spec_tex, it.uv_coord, k_s);
        return -reflection(indir, normal, in_dot_n);
    }
};