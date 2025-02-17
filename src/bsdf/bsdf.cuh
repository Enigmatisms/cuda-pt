/**
 * CUDA BSDF base class implementation
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include <array>
#include <cuda_runtime.h>
#include "core/vec4.cuh"
#include "core/enums.cuh"
#include "core/sampling.cuh"
#include "core/interaction.cuh"

extern const std::array<const char*, NumSupportedBSDF> BSDF_NAMES;

class BSDF {
public:
    Vec4 k_d;
    Vec4 k_s;
    Vec4 k_g;
    int bsdf_flag;
    int __padding;
public:
    CPT_CPU_GPU BSDF() {}
    CPT_CPU_GPU BSDF(Vec4 _k_d, Vec4 _k_s, Vec4 _k_g, int flag = ScatterStateFlag::BSDF_NONE):
        k_d(std::move(k_d)), k_s(std::move(_k_s)), k_g(std::move(_k_g)), bsdf_flag(flag)
    {}

    CPT_GPU void set_kd(Vec4&& v) noexcept { this->k_d = v; }
    CPT_GPU void set_ks(Vec4&& v) noexcept { this->k_s = v; }
    CPT_GPU void set_kg(Vec4&& v) noexcept { this->k_g = v; }
    CPT_GPU void set_lobe(int v) noexcept { this->bsdf_flag = v; }

    CPT_GPU virtual float pdf(const Interaction& it, const Vec3& out, const Vec3& in, int index) const = 0;

    CPT_GPU virtual Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = false) const = 0;

    CPT_GPU virtual Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, ScatterStateFlag& samp_lobe, int index, bool is_radiance = false
    ) const = 0;

    CPT_GPU_INLINE bool require_lobe(ScatterStateFlag flags) const noexcept {
        return (bsdf_flag & (int)flags) > 0;
    }
};


#include "bsdf/bsdf_registry.cuh"