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
 * @brief Phase function base definition
 * @date 2025.02.05
 */

#pragma once

#include "core/enums.cuh"
#include "core/sampler.cuh"
#include "core/vec3.cuh"
#include "core/vec4.cuh"

// POD: phase function sampling sample
struct PhaseSample {
    Vec3 outdir;
    float weight; // phase value / PDF (usually 1)
};

class PhaseFunction {
  public:
    CPT_CPU_GPU PhaseFunction() {}
    CPT_CPU_GPU virtual ~PhaseFunction() {}
    CPT_GPU virtual float eval(Vec3 &&indir, Vec3 &&outdir) const { return 0; }
    // Note that phase function only samples local direction, so
    // the transform from local to world is needed
    CPT_GPU virtual PhaseSample sample(Sampler &sp, Vec3 indir) const {
        return {std::move(indir), 1};
    }

    CPT_GPU virtual void set_param(Vec4 &&data) { ; }
};

extern const std::array<const char *, NumSupportedPhase> PHASES_NAMES;
