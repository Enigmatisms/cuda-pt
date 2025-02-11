#pragma once
/**
 * @file phase.cuh
 * @author Qianyue He
 * @brief Phase function base definition
 * @version 0.1
 * @date 2025-02-05
 * @copyright Copyright (c) 2025
 * 
 */

#include "core/vec3.cuh"
#include "core/sampler.cuh"

// POD: phase function sampling sample
struct PhaseSample {
    Vec3 outdir;
    float weight;           // phase value / PDF (usually 1)
};

class PhaseFunction {
public:
    CPT_CPU_GPU PhaseFunction() {}
    CPT_CPU_GPU virtual ~PhaseFunction() {}
    CPT_GPU virtual float eval(Vec3&& indir, Vec3&& outdir) const {
        return 0;
    }
    // Note that phase function only samples local direction, so
    // the transform from local to world is needed
    CPT_GPU virtual PhaseSample sample(Sampler& sp, Vec3 indir) const {
        return {std::move(indir), 1};
    } 
};