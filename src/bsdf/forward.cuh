/**
 * @file forward.cuh
 * @author Qianyue He
 * @brief Forward BSDF that does not influence radiance
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once

#include "bsdf/bsdf.cuh"
#include "core/textures.cuh"

class ForwardBSDF: public BSDF {
public:
    using BSDF::k_d;
    using BSDF::bsdf_flag;
    CPT_CPU_GPU ForwardBSDF(int flag):
        BSDF(Vec4(0, 1), Vec4(0, 1), Vec4(0, 1), flag) {}

    CPT_CPU_GPU ForwardBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& in, int index) const override {
        return static_cast<float>(in.dot(out) == 1.f);
    }

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override {
        return Vec4(in.dot(out) == 1.f, 1);
    }

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, ScatterStateFlag& samp_lobe, int index, bool is_radiance = true
    ) const override {
        pdf = 1.f;
        samp_lobe = static_cast<ScatterStateFlag>(bsdf_flag);
        return indir;
    }
};