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
 * @author Qianyue He
 * @brief Henyey Greenstein phase function
 * @date 2025.02.05
 */
#pragma once
#include "core/constants.cuh"
#include "core/phase.cuh"
#include "core/sampling.cuh"

class IsotropicPhase : public PhaseFunction {
  public:
    CPT_CPU_GPU IsotropicPhase() {}

    CPT_GPU_INLINE float eval(Vec3 &&indir, Vec3 &&outdir) const override {
        return M_1_Pi * 0.25f;
    }

    CPT_GPU PhaseSample sample(Sampler &sp, Vec3 indir) const override {
        float dummy = 0;
        return {sample_uniform_sphere(sp.next2D(), dummy), 1.f};
    }
};

class HenyeyGreensteinPhase : public PhaseFunction {
  private:
    float g, g2;

  public:
    CPT_CPU_GPU HenyeyGreensteinPhase(float _g) : g(_g), g2(_g * _g) {}

    CPT_GPU static float hg_phase(float cos_theta, float g, float g2) {
        float denom = 1.f + g2 - 2.f * g * cos_theta;
        return M_1_Pi * 0.25f * (1.f - g2) / denom * rsqrtf(denom);
    }

    CPT_GPU float eval(Vec3 &&indir, Vec3 &&outdir) const override {
        float dot_cos = indir.dot(outdir);
        return HenyeyGreensteinPhase::hg_phase(dot_cos, g, g2);
    }

    CPT_GPU PhaseSample sample(Sampler &sp, Vec3 indir) const override {
        Vec2 uv = sp.next2D();
        float sqr_term = (1.f - g2) / (1.f + g - 2.f * g * uv.x());
        float cos_theta = (1.f + g2 - sqr_term * sqr_term) / (2.f * g);
        float sin_theta = sqrtf(fmaxf(0, 1 - cos_theta * cos_theta));
        float sin_phi = 0, cos_phi = 0;
        sincospif(2.f * uv.y(), &sin_phi, &cos_phi);
        return {Vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta), 1.f};
    }

    CPT_GPU void set_param(Vec4 &&data) override {
        g = data.x();
        g2 = data.x() * data.x();
    }
};

class MixedHGPhaseFunction : public PhaseFunction {
  private:
    HenyeyGreensteinPhase ph1;
    HenyeyGreensteinPhase ph2;
    float weight; // the weight for the first phase function
  public:
    CPT_CPU_GPU MixedHGPhaseFunction(float g1, float g2, float w = 0.5)
        : ph1(g1), ph2(g2), weight(w) {}

    CPT_GPU float eval(Vec3 &&indir, Vec3 &&outdir) const override {
        Vec3 temp_indirt = indir, temp_outdir = outdir;
        return ph1.eval(std::move(temp_indirt), std::move(temp_outdir)) *
                   weight +
               ph2.eval(std::move(indir), std::move(outdir)) * (1.f - weight);
    };

    CPT_GPU PhaseSample sample(Sampler &sp, Vec3 indir) const override {
        // MIS
        PhaseSample sp1 = ph1.sample(sp, indir), sp2 = ph2.sample(sp, indir);
        Vec3 dir1 = sp1.outdir, dir2 = sp2.outdir;
        float pdf1 = ph1.eval(Vec3(0, 0, 1), std::move(dir1));
        float pdf2 = ph2.eval(Vec3(0, 0, 1), std::move(dir2));
        bool use_first = sp.next1D() < weight;
        float mis_w = use_first ? pdf1 : pdf2;
        mis_w /= weight * pdf1 + (1.f - weight) * pdf2;
        return {use_first ? dir1 : dir2, mis_w};
    }

    CPT_GPU void set_param(Vec4 &&data) override {
        ph1.set_param(Vec4(data.x()));
        ph2.set_param(Vec4(data.y()));
        weight = data.z();
    }
};
