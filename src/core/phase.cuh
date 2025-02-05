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
    CPT_GPU virtual float eval(Vec3&& indir, Vec3&& outdir) const = 0; 
    // Note that phase function only samples local direction, so
    // the transform from local to world is needed
    CPT_GPU virtual PhaseSample sample(Sampler& sp, Vec3 indir) const = 0; 
};

/**
 * Mixing two phase functions, with MIS
 */
class MixedPhaseFunction: public PhaseFunction {
private:
    PhaseFunction* ph1;
    PhaseFunction* ph2;
    float weight;           // the weight for the first phase function
public:

    CPT_GPU float eval(Vec3&& indir, Vec3&& outdir) const override {
        Vec3 temp_indirt = indir, temp_outdir = outdir;
        return ph1->eval(std::move(temp_indirt), std::move(temp_outdir)) * weight + 
               ph2->eval(std::move(indir), std::move(outdir)) * (1.f - weight);
    };

    CPT_GPU PhaseSample sample(Sampler& sp, Vec3 indir) const override {
        // MIS
        PhaseSample sp1 = ph1->sample(sp, indir),
                    sp2 = ph2->sample(sp, indir);
        Vec3 dir1 = sp1.outdir, dir2 = sp2.outdir;
        float pdf1 = ph1->eval(Vec3(0, 0, 1), std::move(dir1));
        float pdf2 = ph2->eval(Vec3(0, 0, 1), std::move(dir2));
        bool use_first = sp.next1D() < weight;
        float mis_w = use_first ? pdf1 : pdf2;
        mis_w /= weight * pdf1 + (1.f - weight) * pdf2;
        return {use_first ? dir1 : dir2, mis_w};
    } 
};