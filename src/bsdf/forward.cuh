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
 * @file forward.cuh
 * @author Qianyue He
 * @brief Forward BSDF that does not influence radiance
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "bsdf/bsdf.cuh"
#include "core/textures.cuh"

class ForwardBSDF : public BSDF {
  public:
    using BSDF::bsdf_flag;
    using BSDF::k_d;
    CPT_CPU_GPU ForwardBSDF(int flag)
        : BSDF(Vec4(0, 1), Vec4(0, 1), Vec4(0, 1), flag) {}

    CPT_CPU_GPU ForwardBSDF() : BSDF() {}

    CPT_GPU float pdf(const Interaction &it, const Vec3 &out, const Vec3 &in,
                      int index) const override {
        return static_cast<float>(in.dot(out) == 1.f);
    }

    CPT_GPU Vec4 eval(const Interaction &it, const Vec3 &out, const Vec3 &in,
                      int index, bool is_mi = false,
                      bool is_radiance = true) const override {
        return Vec4(in.dot(out) == 1.f, 1);
    }

    CPT_GPU Vec3 sample_dir(const Vec3 &indir, const Interaction &it,
                            Vec4 &throughput, float &pdf, Sampler &sp,
                            ScatterStateFlag &samp_lobe, int index,
                            bool is_radiance = true) const override {
        pdf = 1.f;
        samp_lobe = static_cast<ScatterStateFlag>(bsdf_flag);
        return indir;
    }
};
