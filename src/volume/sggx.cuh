#pragma once
/**
 * @file sggx.cuh
 * @author Qianyue He
 * @brief SGGX phase function (not yet implemented, placeholder here)
 * @version 0.1
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 */

#include "core/phase.cuh"
#include "core/sampling.cuh"
#include "core/constants.cuh"

/**
 * TODO: implement this in the future. Currently, this
 * class works the same as IsotropicPhase
 */
class SGGXPhase: public PhaseFunction {
public:
    CPT_CPU_GPU SGGXPhase() {}

    CPT_GPU_INLINE float eval(Vec3&& indir, Vec3&& outdir) const override {
        return M_1_Pi * 0.25f;
    } 

    CPT_GPU PhaseSample sample(Sampler& sp, Vec3 indir) const override {
        float dummy = 0;
        return {sample_uniform_sphere(sp.next2D(), dummy), 1.f};
    }
};