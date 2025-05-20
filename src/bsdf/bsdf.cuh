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
 * @brief CUDA BSDF base class implementation
 * @date 2024.5.20
 */
#pragma once
#include "core/enums.cuh"
#include "core/interaction.cuh"
#include "core/sampling.cuh"
#include "core/vec4.cuh"
#include <array>
#include <cuda_runtime.h>

extern const std::array<const char *, NumSupportedBSDF> BSDF_NAMES;

class BSDF {
  public:
    Vec4 k_d;
    Vec4 k_s;
    Vec4 k_g;
    int bsdf_flag;
    int __padding;

  public:
    CPT_CPU_GPU BSDF() {}
    CPT_CPU_GPU BSDF(Vec4 _k_d, Vec4 _k_s, Vec4 _k_g,
                     int flag = ScatterStateFlag::BSDF_NONE)
        : k_d(std::move(k_d)), k_s(std::move(_k_s)), k_g(std::move(_k_g)),
          bsdf_flag(flag) {}

    CPT_GPU void set_kd(Vec4 &&v) noexcept { this->k_d = v; }
    CPT_GPU void set_ks(Vec4 &&v) noexcept { this->k_s = v; }
    CPT_GPU void set_kg(Vec4 &&v) noexcept { this->k_g = v; }
    CPT_GPU void set_lobe(int v) noexcept { this->bsdf_flag = v; }

    CPT_GPU virtual float pdf(const Interaction &it, const Vec3 &out,
                              const Vec3 &in, int index) const = 0;

    CPT_GPU virtual Vec4 eval(const Interaction &it, const Vec3 &out,
                              const Vec3 &in, int index, bool is_mi = false,
                              bool is_radiance = false) const = 0;

    CPT_GPU virtual Vec3 sample_dir(const Vec3 &indir, const Interaction &it,
                                    Vec4 &throughput, float &pdf, Sampler &sp,
                                    ScatterStateFlag &samp_lobe, int index,
                                    bool is_radiance = false) const = 0;

    CPT_GPU_INLINE bool require_lobe(ScatterStateFlag flags) const noexcept {
        return (bsdf_flag & (int)flags) > 0;
    }
};

#include "bsdf/bsdf_registry.cuh"
