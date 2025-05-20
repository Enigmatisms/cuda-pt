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

#pragma once
/**
 * @file sggx.cuh
 * @author Qianyue He
 * @brief SGGX phase function (not yet implemented, placeholder here)
 * @version 0.1
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 */

#include "core/constants.cuh"
#include "core/phase.cuh"
#include "core/sampling.cuh"

/**
 * TODO: implement this in the future. Currently, this
 * class works the same as IsotropicPhase
 */
class SGGXPhase : public PhaseFunction {
  public:
    CPT_CPU_GPU SGGXPhase() {}

    CPT_GPU_INLINE float eval(Vec3 &&indir, Vec3 &&outdir) const override {
        return M_1_Pi * 0.25f;
    }

    CPT_GPU PhaseSample sample(Sampler &sp, Vec3 indir) const override {
        float dummy = 0;
        return {sample_uniform_sphere(sp.next2D(), dummy), 1.f};
    }
};
