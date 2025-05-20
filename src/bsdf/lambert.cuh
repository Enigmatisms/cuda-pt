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
 * @brief Lambertian BSDF
 * @date 2025.01.06
 */
#pragma once

#include "bsdf/bsdf.cuh"
#include "core/textures.cuh"

class LambertianBSDF : public BSDF {
  public:
    using BSDF::bsdf_flag;
    using BSDF::k_d;
    CPT_CPU_GPU LambertianBSDF(Vec4 _k_d, int kd_id = -1)
        : BSDF(std::move(_k_d), Vec4(0, 0, 0), Vec4(0, 0, 0),
               ScatterStateFlag::BSDF_DIFFUSE |
                   ScatterStateFlag::BSDF_REFLECT) {}

    CPT_CPU_GPU LambertianBSDF() : BSDF() {}

    CPT_GPU float pdf(const Interaction &it, const Vec3 &out,
                      const Vec3 & /* in */, int index) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        float dot_val = normal.dot(out);
        return max(normal.dot(out), 0.f) * M_1_Pi;
    }

    CPT_GPU Vec4 eval(const Interaction &it, const Vec3 &out, const Vec3 &in,
                      int index, bool is_mi = false,
                      bool is_radiance = true) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        float cos_term = normal.dot(out);
        float dot_in = normal.dot(in);
        float same_side =
            (dot_in > 0) ^
            (cos_term > 0); // should be positive or negative at the same time
        const cudaTextureObject_t diff_tex = c_textures.diff_tex[index];
        return c_textures.eval(diff_tex, it.uv_coord, k_d) *
               max(0.f, cos_term) * M_1_Pi * same_side;
    }

    CPT_GPU Vec3 sample_dir(const Vec3 &indir, const Interaction &it,
                            Vec4 &throughput, float &pdf, Sampler &sp,
                            ScatterStateFlag &samp_lobe, int index,
                            bool is_radiance = true) const override {
        auto local_ray = sample_cosine_hemisphere(sp.next2D(), pdf);
        const Vec3 normal = c_textures.eval_normal(it, index);
        auto out_ray = delocalize_rotate(normal, local_ray);
        // throughput *= f / pdf --> k_d * cos / pi / (pdf = cos / pi) == k_d
        float dot_in = normal.dot(indir);
        float dot_out = normal.dot(out_ray);
        const cudaTextureObject_t diff_tex = c_textures.diff_tex[index];
        throughput *= c_textures.eval(diff_tex, it.uv_coord, k_d) *
                      ((dot_in > 0) ^ (dot_out > 0));
        samp_lobe = static_cast<ScatterStateFlag>(bsdf_flag);
        return out_ray;
    }
};
