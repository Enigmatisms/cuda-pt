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
 * @file plastic.cuh
 * @author Qianyue He
 * @brief Plastic and PlasticForward BSDF
 * @date 2025-01-06
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "bsdf/bsdf.cuh"
#include "bsdf/fresnel.cuh"
#include "core/textures.cuh"

class PlasticBSDF : public BSDF {
  public:
    float trans_scaler;
    float thickness;
    float eta;
    float precomp_diff_f; // precomputed diffuse Fresnel

    using BSDF::k_d;
    using BSDF::k_g;
    using BSDF::k_s;

  public:
    // Penetration: if true, the plastic material will allow 'erroneous' light
    // leaks (light leak: incident ray and exiting ray are not in the same
    // hemisphere) if light leak is enabled, the plastic material is more
    // suitable for plants
    CPT_CPU_GPU PlasticBSDF(Vec4 _k_d, Vec4 _k_s, Vec4 sigma_a, float ior,
                            float trans_scaler = 1.f, float thickness = 0,
                            bool penetration = false);

    CPT_CPU_GPU PlasticBSDF() : BSDF() {}

    CPT_GPU float pdf(const Interaction &it, const Vec3 &out,
                      const Vec3 & /* in */, int) const override;

    CPT_GPU Vec4 eval(const Interaction &it, const Vec3 &out, const Vec3 &in,
                      int index, bool is_mi = false,
                      bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(const Vec3 &indir, const Interaction &it,
                            Vec4 &throughput, float &pdf, Sampler &sp,
                            ScatterStateFlag &samp_lobe, int index,
                            bool is_radiance = true) const override;
};

/**
 * @brief specular reflection and delta forward
 */
class PlasticForwardBSDF : public BSDF {
  public:
    float trans_scaler;
    float thickness;
    float eta;

    using BSDF::k_d;
    using BSDF::k_g;
    using BSDF::k_s;

  public:
    // the last parameter is a dummy parameter (thus we can have the same API
    // with PlasticBSDF)
    CPT_CPU_GPU PlasticForwardBSDF(Vec4 _k_d, Vec4 _k_s, Vec4 sigma_a,
                                   float ior, float trans_scaler = 1.f,
                                   float thickness = 0, bool _dummy = false);

    CPT_CPU_GPU PlasticForwardBSDF() : BSDF() {}

    CPT_GPU float pdf(const Interaction &it, const Vec3 &out,
                      const Vec3 & /* in */, int index) const override;

    CPT_GPU Vec4 eval(const Interaction &it, const Vec3 &out, const Vec3 &in,
                      int index, bool is_mi = false,
                      bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(const Vec3 &indir, const Interaction &it,
                            Vec4 &throughput, float &pdf, Sampler &sp,
                            ScatterStateFlag &samp_lobe, int index,
                            bool is_radiance = true) const override;
};
