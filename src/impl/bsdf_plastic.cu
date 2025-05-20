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
 * @brief Plastic BSDF model
 * @date 2024.11.06
 */
#include "bsdf/plastic.cuh"

CPT_CPU_GPU PlasticBSDF::PlasticBSDF(Vec4 _k_d, Vec4 _k_s, Vec4 sigma_a,
                                     float ior, float trans_scaler,
                                     float thickness, bool penetration)
    : BSDF(_k_d, std::move(_k_s), std::move(sigma_a),
           ScatterStateFlag::BSDF_SPECULAR | ScatterStateFlag::BSDF_DIFFUSE |
               ScatterStateFlag::BSDF_REFLECT),
      trans_scaler(trans_scaler), thickness(thickness), eta(1.f / ior) {
    precomp_diff_f = FresnelTerms::diffuse_fresnel(ior);
    __padding = penetration;
}

CPT_GPU float PlasticBSDF::pdf(const Interaction &it, const Vec3 &out,
                               const Vec3 &in, int index) const {
    const Vec3 normal = c_textures.eval_normal(it, index);
    float dot_wo = fabs(out.dot(normal)), dot_wi = fabs(in.dot(normal)),
          Fi = FresnelTerms::fresnel_simple(eta, dot_wi),
          specular_prob = Fi / (Fi + trans_scaler * (1.0f - Fi));
    Vec3 refdir = -reflection(in, normal);
    // 'refdir.dot(out) >= 1.f - THP_EPS' means reflected incident ray is very
    // close to 'out'
    float pdf = refdir.dot(out) < 1.f - THP_EPS
                    ? M_1_Pi * dot_wo * (1.f - specular_prob)
                    : specular_prob;
    return pdf;
}

CPT_GPU Vec4 PlasticBSDF::eval(const Interaction &it, const Vec3 &out,
                               const Vec3 &in, int index, bool is_mi,
                               bool is_radiance) const {
    const Vec3 normal = c_textures.eval_normal(it, index);
    float raw_dot_wo = out.dot(normal), raw_dot_wi = in.dot(normal);
    float dot_wo = fabsf(raw_dot_wo), dot_wi = fabsf(raw_dot_wi),
          Fi = FresnelTerms::fresnel_simple(eta, dot_wi),
          Fo = FresnelTerms::fresnel_simple(eta, dot_wo);
    Vec3 refdir = -reflection(in, normal);

    const cudaTextureObject_t
        diff_tex = c_textures.diff_tex[index],
        spec_tex = c_textures.spec_tex[index],
        siga_tex =
            c_textures.glos_tex[index]; // sigma_a is stored in glossy texture
    Vec4 k_diff = c_textures.eval(diff_tex, it.uv_coord, k_d);
    Vec4 brdf = (M_1_Pi * (1.0f - Fi) * (1.0f - Fo) * eta * eta) * dot_wo *
                (k_diff / (-k_diff * precomp_diff_f + 1.f)) *
                (c_textures.eval(siga_tex, it.uv_coord, k_g) * thickness *
                 (-1.0f / dot_wo - 1.0f / dot_wi))
                    .exp_xyz();
    brdf += refdir.dot(out) < 1.f - THP_EPS
                ? Vec4(0)
                : Vec4(Fi, 1) * c_textures.eval(spec_tex, it.uv_coord, k_s);
    brdf = (__padding > 0 || (raw_dot_wo > 0) ^ (raw_dot_wi > 0)) ? brdf
                                                                  : Vec4(0, 1);
    return brdf;
}

CPT_GPU Vec3 PlasticBSDF::sample_dir(const Vec3 &indir, const Interaction &it,
                                     Vec4 &throughput, float &pdf, Sampler &sp,
                                     ScatterStateFlag &samp_lobe, int index,
                                     bool is_radiance) const {
    const Vec3 normal = c_textures.eval_normal(it, index);
    float raw_dot_indir = normal.dot(indir), dot_indir = fabsf(raw_dot_indir);
    ;
    float Fi = FresnelTerms::fresnel_simple(eta, dot_indir),
          specular_prob = Fi / (Fi + trans_scaler * (1.0f - Fi));

    Vec3 outdir;

    if (sp.next1D() < specular_prob) { // coating specular reflection
        outdir = -reflection(indir, normal);
        pdf = specular_prob;
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        throughput *= Vec4(Fi / specular_prob, 1) *
                      c_textures.eval(spec_tex, it.uv_coord, k_s);
        samp_lobe = static_cast<ScatterStateFlag>(
            ScatterStateFlag::BSDF_REFLECT | ScatterStateFlag::BSDF_SPECULAR);
    } else { // substrate diffuse reflection
        float dummy_v = 1;
        Vec3 local_dir = sample_cosine_hemisphere(sp.next2D(), dummy_v);
        float Fo = FresnelTerms::fresnel_simple(eta, local_dir.z());

        const cudaTextureObject_t
            diff_tex = c_textures.diff_tex[index],
            siga_tex =
                c_textures
                    .glos_tex[index]; // sigma_a is stored in glossy texture
        Vec4 k_diff = c_textures.eval(diff_tex, it.uv_coord, k_d);

        // use fmaxf instead of (1-Fi) * (1-Fo), to make the result looks
        // brighter
        Vec4 local_thp =
            ((1.0f - Fi) * (1.0f - Fo) * eta * eta) *
            (k_diff / (-k_diff * precomp_diff_f + 1.f)) *
            (c_textures.eval(siga_tex, it.uv_coord, k_g) * thickness *
             (-1.0f / local_dir.z() - 1.0f / dot_indir))
                .exp_xyz();

        pdf = M_1_Pi * local_dir.z() * (1.0f - specular_prob);
        throughput *= __frcp_rn(1.f - specular_prob) * local_thp;
        outdir = delocalize_rotate(normal, local_dir);
        samp_lobe = static_cast<ScatterStateFlag>(
            ScatterStateFlag::BSDF_REFLECT | ScatterStateFlag::BSDF_DIFFUSE);
    }
    throughput =
        (__padding > 0 || (raw_dot_indir > 0) ^ (outdir.dot(normal) > 0))
            ? throughput
            : Vec4(0, 1);
    return outdir;
}

// ===================== plastic forward =======================

CPT_CPU_GPU PlasticForwardBSDF::PlasticForwardBSDF(Vec4 _k_d, Vec4 _k_s,
                                                   Vec4 sigma_a, float ior,
                                                   float trans_scaler,
                                                   float thickness, bool)
    : BSDF(_k_d, std::move(_k_s), std::move(sigma_a),
           ScatterStateFlag::BSDF_SPECULAR | ScatterStateFlag::BSDF_TRANSMIT |
               ScatterStateFlag::BSDF_REFLECT),
      trans_scaler(trans_scaler), thickness(thickness), eta(1.f / ior) {}

CPT_GPU float PlasticForwardBSDF::pdf(const Interaction &it, const Vec3 &out,
                                      const Vec3 &in, int index) const {
    const Vec3 normal = c_textures.eval_normal(it, index);
    float dot_wi = in.dot(normal),
          Fi = FresnelTerms::fresnel_simple(eta, -dot_wi),
          specular_prob = Fi / (Fi + trans_scaler * (1.0f - Fi));
    Vec3 refdir = -reflection(in, normal);
    // 'refdir.dot(out) >= 1.f - THP_EPS' means reflected incident ray is very
    // close to 'out'
    float pdf = 0;
    pdf = refdir.dot(out) < 1.f - THP_EPS ? pdf : specular_prob;
    pdf = in.dot(out) < 1.f - THP_EPS ? pdf : 1.f - specular_prob;
    return pdf;
}

CPT_GPU Vec4 PlasticForwardBSDF::eval(const Interaction &it, const Vec3 &out,
                                      const Vec3 &in, int index, bool is_mi,
                                      bool is_radiance) const {
    const Vec3 normal = c_textures.eval_normal(it, index);
    float dot_wi = in.dot(normal),
          Fi = FresnelTerms::fresnel_simple(eta, fabsf(dot_wi));
    Vec3 refdir = -reflection(in, normal);

    const cudaTextureObject_t
        diff_tex = c_textures.diff_tex[index],
        spec_tex = c_textures.spec_tex[index],
        siga_tex =
            c_textures.glos_tex[index]; // sigma_a is stored in glossy texture
    Vec4 brdf = in.dot(out) < 1.f - THP_EPS
                    ? Vec4(0, 1)
                    : (1.0f - Fi) * (1.0f - Fi) *
                          c_textures.eval(diff_tex, it.uv_coord, k_d) * eta *
                          eta *
                          (c_textures.eval(siga_tex, it.uv_coord, k_g) *
                           thickness * (-2.f / fabsf(dot_wi)))
                              .exp_xyz(); // transmit?
    brdf += refdir.dot(out) < 1.f - THP_EPS
                ? brdf
                : Vec4(Fi, 1) * c_textures.eval(spec_tex, it.uv_coord, k_s);
    return brdf;
}

CPT_GPU Vec3 PlasticForwardBSDF::sample_dir(const Vec3 &indir,
                                            const Interaction &it,
                                            Vec4 &throughput, float &pdf,
                                            Sampler &sp,
                                            ScatterStateFlag &samp_lobe,
                                            int index, bool is_radiance) const {
    const Vec3 normal = c_textures.eval_normal(it, index);
    float dot_indir = normal.dot(indir),
          Fi = FresnelTerms::fresnel_simple(eta, fabsf(dot_indir)),
          specular_prob = Fi / (Fi + trans_scaler * (1.0f - Fi));

    Vec3 outdir;

    if (sp.next1D() < specular_prob) { // coating specular reflection
        outdir = -reflection(indir, normal, dot_indir);
        pdf = specular_prob;
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        throughput *= Vec4(Fi / specular_prob, 1) *
                      c_textures.eval(spec_tex, it.uv_coord, k_s);
        samp_lobe = static_cast<ScatterStateFlag>(
            ScatterStateFlag::BSDF_REFLECT | ScatterStateFlag::BSDF_SPECULAR);
    } else { // substrate diffuse reflection
        const cudaTextureObject_t
            diff_tex = c_textures.diff_tex[index],
            siga_tex =
                c_textures
                    .glos_tex[index]; // sigma_a is stored in glossy texture
        Vec4 local_thp = ((1.0f - Fi) * (1.0f - Fi) * eta * eta) *
                         c_textures.eval(diff_tex, it.uv_coord, k_d) *
                         (c_textures.eval(siga_tex, it.uv_coord, k_g) *
                          thickness * (-2.0f / fabsf(dot_indir)))
                             .exp_xyz();

        pdf = 1.0f - specular_prob;
        throughput *= __frcp_rn(1.f - specular_prob) * local_thp;

        outdir = indir;
        samp_lobe = static_cast<ScatterStateFlag>(
            ScatterStateFlag::BSDF_TRANSMIT | ScatterStateFlag::BSDF_SPECULAR);
    }
    return outdir;
}
