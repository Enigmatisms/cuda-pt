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
 * @file dispersion.cuh
 * @author Qianyue He
 * @brief 360nm to 830nm Wave dispersion translucent BSDF
 * @date 2025-01-06
 * @copyright Copyright (c) 2025
 */

#pragma once
#include "bsdf/translucent.cuh"

/**
 * @brief 360nm to 830nm Wave dispersion translucent BSDF
 * We uniformly sample from wavelength range 360nm to 830nm,
 * and the IoR is computed by Cauchy's Equation A + B / \lambda^2
 */
class DispersionBSDF : public BSDF {
  public:
    static constexpr float WL_MIN = 360;
    static constexpr float WL_RANGE =
        471; // 360 + 471 -> 831 (830 included for indexing)
    static constexpr float D65_MIN = 300;
    static constexpr float D65_RANGE = 531;

  public:
    CPT_CPU_GPU DispersionBSDF(Vec4 k_s, float index_a, float index_b)
        : BSDF(Vec4(index_a, index_b, 0), std::move(k_s), Vec4(0, 0, 0),
               ScatterStateFlag::BSDF_DIFFUSE |
                   ScatterStateFlag::BSDF_TRANSMIT) {}

    CPT_CPU_GPU DispersionBSDF() : BSDF() {}

    CPT_GPU float pdf(const Interaction &it, const Vec3 &out, const Vec3 &incid,
                      int) const override;
    CPT_GPU Vec4 eval(const Interaction &it, const Vec3 &out, const Vec3 &in,
                      int index, bool is_mi = false,
                      bool is_radiance = true) const override;
    CPT_GPU Vec3 sample_dir(const Vec3 &indir, const Interaction &it,
                            Vec4 &throughput, float &pdf, Sampler &sp,
                            ScatterStateFlag &samp_lobe, int index,
                            bool is_radiance = true) const override;

    CPT_GPU_INLINE static Vec4 wavelength_to_XYZ(float wavelength);
    CPT_GPU_INLINE static Vec4 wavelength_to_RGB(float wavelength);

    CPT_GPU_INLINE static float sample_wavelength(Sampler &sp) {
        return sp.next1D() * WL_RANGE + WL_MIN;
    }

    CPT_GPU_INLINE float get_ior(float wavelength) const {
        // k_d.y() is B, (nm^2)
        return k_d.x() + k_d.y() / (wavelength * wavelength);
    }

    CONDITION_TEMPLATE_SEP_3(VType1, VType2, NType, Vec3, Vec3, Vec3)
    CPT_GPU_INLINE bool get_wavelength_from(VType1 &&indir, VType2 &&outdir,
                                            NType &&normal,
                                            float &wavelength) const {
        float cos_i = normal.dot(indir), cos_o = normal.dot(outdir),
              sin_i = sqrtf(1.f - cos_i * cos_i),
              sin_o = sqrtf(1.f - cos_o * cos_o);
        float eta = sin_i > sin_o ? sin_i / sin_o : sin_o / sin_i;
        wavelength = sqrtf(k_d.y() / fmaxf(eta - k_d.x(), 1e-5f));
        return wavelength > WL_MIN && wavelength < WL_MIN + WL_RANGE;
    }
};
