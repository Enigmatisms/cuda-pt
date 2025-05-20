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
 * @file ggx_conductor.cuh
 * @author Qianyue He
 * @brief GGX microfacet normal distribution based BSDF
 * @date 2025-01-06
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "bsdf/bsdf.cuh"
#include "bsdf/fresnel.cuh"
#include "core/textures.cuh"

class GGXConductorBSDF : public BSDF {
    /**
     * @brief GGX microfacet normal distribution based BSDF
     * k_d is the eta_t of the metal
     * k_s is the k (Vec3) and the mapped roughness (k_s[3])
     * k_g is the underlying color (albedo)
     */
  public:
    using BSDF::k_s;
    FresnelTerms fresnel;

  public:
    CPT_CPU_GPU GGXConductorBSDF(Vec3 eta_t, Vec3 k, Vec4 albedo,
                                 float roughness_x, float roughness_y);

    CPT_CPU_GPU GGXConductorBSDF() : BSDF() {}

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
