/**
 * @file rayleigh.cuh
 * @author Qianyue He
 * @brief Rayleigh phase function
 * @version 0.1
 * @date 2025-02-05
 * @copyright Copyright (c) 2025
 * 
 * Implement the direct inverse CDF from paper:
 * Importance Sampling the Rayleigh Phase Function
 */

#include "core/phase.cuh"
#include "core/sampling.cuh"
#include "core/constants.cuh"

class RayleighPhase: public PhaseFunction {
public:
    CPT_CPU_GPU RayleighPhase() {}

    CPT_GPU_INLINE float eval(Vec3&& indir, Vec3&& outdir) const override {
        return 3.f / 16.f * M_1_Pi * (1.f + indir.dot(outdir));
    } 

    CPT_GPU PhaseSample sample(Sampler& sp, Vec3 indir) const override {
        Vec2 uv = sp.next2D();
        float u_rs = uv.x() * 2.f - 1.f;
        float cos_theta = cbrtf(2.f * u_rs + sqrtf(4.f * u_rs * u_rs + 1.f)); 
        float sin_theta = sqrtf(fmaxf(0, 1 - cos_theta * cos_theta));
        float sin_phi = 0, cos_phi = 0;
        sincospif(2.f * uv.y(), &sin_phi, &cos_phi);
        // this is analytical importance sampling, so the weight is 1
        return {Vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta), 1.f};
    }
};