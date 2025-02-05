#pragma once
/**
 * @file henyey_greenstein.cuh
 * @author Qianyue He
 * @brief Henyey Greenstein phase function
 * @version 0.1
 * @date 2025-02-05
 * @copyright Copyright (c) 2025
 */
#include "core/phase.cuh"
#include "core/sampling.cuh"
#include "core/constants.cuh"

class IsotropicPhase: public PhaseFunction {
public:
    CPT_CPU_GPU IsotropicPhase() {}

    CPT_GPU_INLINE float eval(Vec3&& indir, Vec3&& outdir) const override {
        return M_1_Pi * 0.25f;
    } 

    CPT_GPU PhaseSample sample(Sampler& sp, Vec3 indir) const override {
        float dummy = 0;
        return {sample_uniform_sphere(sp.next2D(), dummy), 1.f};
    }
};

class HenyeyGreensteinPhase: public PhaseFunction {
private:
    float g, g2;
public:
    CPT_CPU_GPU HenyeyGreensteinPhase(float _g): g(_g), g2(_g * _g) {}

    CPT_GPU static float hg_phase(float cos_theta, float g, float g2) {
        float denom = 1.f + g2 - 2.f * g * cos_theta;
        return M_1_Pi * 0.25f * (1.f - g2) / denom * rsqrtf(denom);
    }

    CPT_GPU float eval(Vec3&& indir, Vec3&& outdir) const override {
        float dot_cos = indir.dot(outdir);
        return HenyeyGreensteinPhase::hg_phase(dot_cos, g, g2);
    } 

    CPT_GPU PhaseSample sample(Sampler& sp, Vec3 indir) const override {
        Vec2 uv = sp.next2D();
        float sqr_term = (1.f - g2) / (1.f + g - 2.f * g * uv.x());
        float cos_theta = (1.f + g2 - sqr_term * sqr_term) / (2.f * g);
        float sin_theta = sqrtf(fmaxf(0, 1 - cos_theta * cos_theta));
        float sin_phi = 0, cos_phi = 0;
        sincospif(2.f * uv.y(), &sin_phi, &cos_phi);
        return {Vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta), 1.f};
    } 
};