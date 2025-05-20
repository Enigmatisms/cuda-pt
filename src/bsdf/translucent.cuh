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
 * @brief Fresnel Translucent BSDF
 * @date 2025.01.06
 *
 */
#pragma once

#include "bsdf/bsdf.cuh"
#include "bsdf/fresnel.cuh"
#include "core/textures.cuh"

class TranslucentBSDF : public BSDF {
  public:
    using BSDF::k_d; // ior
    using BSDF::k_s; // specular reflection

    CPT_CPU_GPU TranslucentBSDF(Vec4 k_s, Vec4 ior)
        : BSDF(std::move(ior), std::move(k_s), Vec4(0, 0, 0),
               ScatterStateFlag::BSDF_SPECULAR |
                   ScatterStateFlag::BSDF_TRANSMIT) {}

    CPT_CPU_GPU TranslucentBSDF() : BSDF() {}

    CPT_GPU float pdf(const Interaction &it, const Vec3 &out, const Vec3 &incid,
                      int) const override {
        return 0.f;
    }

    CPT_GPU_INLINE static Vec4 eval_impl(const Vec3 &normal, const Vec3 &out,
                                         const Vec3 &in, const Vec4 &ks,
                                         const float eta,
                                         bool is_radiance = false) {
        float dot_normal = in.dot(normal);
        // at least according to pbrt-v3, ni / nr is computed as the following
        // (using shading normal) see
        // https://computergraphics.stackexchange.com/questions/13540/shading-normal-and-geometric-normal-for-refractive-surface-rendering
        float ni = dot_normal < 0 ? 1.f : eta;
        float nr = dot_normal < 0 ? eta : 1.f, cos_r2 = 0;
        float eta2 = ni * ni / (nr * nr);
        Vec3 ret_dir = in.advance(normal, -2.f * dot_normal).normalized(),
             refra_vec = FresnelTerms::snell_refraction(in, normal, cos_r2,
                                                        dot_normal, ni, nr);
        bool total_ref = FresnelTerms::is_total_reflection(dot_normal, ni, nr);
        nr = FresnelTerms::fresnel_dielectric(ni, nr, fabsf(dot_normal),
                                              sqrtf(fabsf(cos_r2)));
        bool reflc_dot = out.dot(ret_dir) > 0.99999f,
             refra_dot =
                 out.dot(refra_vec) > 0.99999f; // 0.9999  means 0.26 deg
        return ks * (reflc_dot | refra_dot) *
               (refra_dot && is_radiance ? eta2 : 1.f);
    }

    CPT_GPU_INLINE static Vec3
    sample_dir_impl(const Vec3 &indir, const Vec3 &normal, const Vec4 &ks,
                    const float eta, Vec4 &throughput, Sampler &sp, float &pdf,
                    ScatterStateFlag &samp_lobe, bool is_radiance = false) {
        float dot_normal = indir.dot(normal);
        // at least according to pbrt-v3, ni / nr is computed as the following
        // (using shading normal) see
        // https://computergraphics.stackexchange.com/questions/13540/shading-normal-and-geometric-normal-for-refractive-surface-rendering
        float ni = dot_normal < 0 ? 1.f : eta;
        float nr = dot_normal < 0 ? eta : 1.f, cos_r2 = 0;
        float eta2 = ni * ni / (nr * nr);
        Vec3 ret_dir = indir.advance(normal, -2.f * dot_normal).normalized(),
             refra_vec = FresnelTerms::snell_refraction(indir, normal, cos_r2,
                                                        dot_normal, ni, nr);
        bool total_ref = FresnelTerms::is_total_reflection(dot_normal, ni, nr);
        nr = FresnelTerms::fresnel_dielectric(ni, nr, fabsf(dot_normal),
                                              sqrtf(fabsf(cos_r2)));
        bool reflect = total_ref || sp.next1D() < nr;
        ret_dir = select(ret_dir, refra_vec, reflect);
        pdf = total_ref ? 1.f : (reflect ? nr : 1.f - nr);
        samp_lobe = static_cast<ScatterStateFlag>(
            ScatterStateFlag::BSDF_SPECULAR |
            (total_ref || reflect ? ScatterStateFlag::BSDF_REFLECT
                                  : ScatterStateFlag::BSDF_TRANSMIT));
        throughput *= ks * (is_radiance && !reflect ? eta2 : 1.f);
        return ret_dir;
    }

    CPT_GPU Vec4 eval(const Interaction &it, const Vec3 &out, const Vec3 &in,
                      int index, bool is_mi = false,
                      bool is_radiance = false) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        const Vec4 ks = c_textures.eval(spec_tex, it.uv_coord, k_s);
        float eta =
            c_textures.eval_rough(it.uv_coord, index, Vec2(k_d.x())).x();
        return eval_impl(normal, out, in, ks, eta, is_radiance);
    }

    CPT_GPU Vec3 sample_dir(const Vec3 &indir, const Interaction &it,
                            Vec4 &throughput, float &pdf, Sampler &sp,
                            ScatterStateFlag &samp_lobe, int index,
                            bool is_radiance = false) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        const Vec4 ks = c_textures.eval(spec_tex, it.uv_coord, k_s);
        float eta =
            c_textures.eval_rough(it.uv_coord, index, Vec2(k_d.x())).x();
        return sample_dir_impl(indir, normal, ks, eta, throughput, sp, pdf,
                               samp_lobe, is_radiance);
    }
};
