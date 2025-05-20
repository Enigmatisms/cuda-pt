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

#include "bsdf/ggx_conductor.cuh"

/**
 * GGX microfacet model
 */
class GGX {
  private:
    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static float get_lambda(VecType &&local, float alphax,
                                           float alphay) {
        float e = GGX::e_func(std::forward<VecType>(local), alphax, alphay);
        return e == 0 ? 0 : (-1.f + sqrtf(1.f + e)) * 0.5f;
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    static CPT_GPU_INLINE float e_func(VecType &&local, float alphax,
                                       float alphay) {
        float cos2_theta = local.z() * local.z(),
              inv_cos2_theta = cos2_theta == 0 ? 0 : __frcp_rn(cos2_theta);
        // avoid calculating cos phi and sin phi
        return (local.x() * local.x() * alphax * alphax +
                local.y() * local.y() * alphay * alphay) *
               inv_cos2_theta;
    }

  public:
    // input cos theta and roughness, with random sample uv, output scaled
    // slopes
    static CPT_GPU void ggx_cos_sample(float cos_theta, Vec2 &&uv,
                                       Vec2 &slopes) {
        if (cos_theta == 1) {
            float r = sqrt(uv.x() / (1 - uv.y()));
            float phi = 2.f * uv.y(), sin_phi = 0, cos_phi = 0;

            sincospif(phi, &sin_phi, &cos_phi);
            slopes.x() = r * cos_phi;
            slopes.y() = r * sin_phi;
            return;
        }

        float sin_theta = sqrtf(fmaxf(0, 1.f - cos_theta * cos_theta));
        float tan_theta = sin_theta / cos_theta;
        float a = 1.f / tan_theta;
        float G1 = 2.f / (1 + sqrtf(1.f + 1.f / (a * a)));

        // sample slope_x
        float A = 2.f * uv.x() / G1 - 1;
        float tmp = fminf(1.f / (A * A - 1.f), 1e9);
        float B = tan_theta;
        float D = sqrtf(fmaxf(B * B * tmp * tmp - (A * A - B * B) * tmp, 0));
        float slope_x_1 = B * tmp - D;
        float slope_x_2 = B * tmp + D;
        slopes.x() =
            (A < 0 || slope_x_2 > 1.f / tan_theta) ? slope_x_1 : slope_x_2;

        // sample slope_y
        float sign = uv.y() > 0.5f ? 1.f : -1.f;
        uv.y() = 2.f * (uv.y() - .5f) * sign;

        float z =
            (uv.y() * (uv.y() * (uv.y() * 0.27385f - 0.73369f) + 0.46341f)) /
            (uv.y() * (uv.y() * (uv.y() * 0.093073f + 0.309420f) - 1.000000f) +
             0.597999f);
        slopes.y() = sign * z * sqrtf(1.f + slopes.x() * slopes.x());
    }

    static CPT_GPU Vec2 get_sincos_phi(const Vec3 &v) {
        float sin_theta2 = fmaxf(1.f - v.z() * v.z(), 0),
              inv_sin_theta = rsqrtf(sin_theta2);
        bool theta_zero = sin_theta2 == 0;
        return Vec2(theta_zero ? 1 : fminf(fmaxf(v.x() * inv_sin_theta, -1), 1),
                    theta_zero ? 0
                               : fminf(fmaxf(v.y() * inv_sin_theta, -1), 1));
    }

    // return Vec2: (D value, e value for reuse)
    CONDITION_TEMPLATE(VecType, Vec3)
    static CPT_GPU_INLINE float D(VecType &&local, float alphax, float alphay) {
        // e can be directly used to calculate G1
        float cos2_theta = local.z() * local.z(),
              inv_cos2_theta = cos2_theta == 0 ? 0 : __frcp_rn(cos2_theta);
        // avoid calculating cos phi and sin phi
        float e = (local.x() * local.x() / (alphax * alphax) +
                   local.y() * local.y() / (alphay * alphay)) *
                  inv_cos2_theta;

        return __frcp_rn(M_Pi * alphax * alphay * cos2_theta * cos2_theta *
                         (1 + e) * (1 + e));
    }

    static CPT_GPU_INLINE float G1_with_e(float e) {
        float lambda = e == 0 ? 0 : (-1.f + sqrtf(1.f + e)) * 0.5f;
        return 1.f / (1.f + lambda);
    }

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    static CPT_GPU_INLINE float G(VType1 &&local_in, VType2 &&local_out,
                                  float alphax, float alphay) {
        return 1.f / (1.f + get_lambda(local_in, alphax, alphay) +
                      get_lambda(local_out, alphax, alphay));
    }

  public:
    /**
     * @brief sample microfacet normal
     * the inputs should be in the local frame
     */
    CONDITION_TEMPLATE(VecType, Vec3)
    static CPT_GPU_INLINE Vec3 sample_wh(VecType &&local_indir, float alphax,
                                         float alphay, Vec2 &&uv) {
        Vec3 wi_stretched = Vec3(local_indir.x() * alphax,
                                 local_indir.y() * alphay, local_indir.z())
                                .normalized();
        Vec2 slope;

        ggx_cos_sample(wi_stretched.z(), std::move(uv), slope);
        Vec2 cos_sin_phi = get_sincos_phi(wi_stretched);

        float tmp = cos_sin_phi.x() * slope.x() - cos_sin_phi.y() * slope.y();
        slope.y() =
            (cos_sin_phi.y() * slope.x() + cos_sin_phi.x() * slope.y()) *
            alphay;
        slope.x() = tmp * alphax;

        return Vec3(-slope.x(), -slope.y(), 1.).normalized();
    }

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    static CPT_GPU_INLINE float pdf(VType1 &&local_wh, VType2 &&local_in,
                                    float alphax, float alphay) {
        auto D_e = D(local_wh, alphax, alphay);
        // can be 0 for the denominator
        float cos_ratio = fabsf(local_in.dot(local_wh)) / fabsf(local_in.z());
        return D_e * G1(std::forward<VType2>(local_in), alphax, alphay) *
               cos_ratio;
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    static CPT_GPU_INLINE float G1(VecType &&local, float alphax,
                                   float alphay) {
        return 1.f / (1.f + get_lambda(local, alphax, alphay));
    }

    // indir points towards the incident point, outdir points away from it
    CONDITION_TEMPLATE_SEP_3(VType1, VType2, VType3, Vec3, Vec3, Vec3)
    static CPT_GPU Vec4 eval(VType1 &&n_s, VType2 &&indir, VType3 &&outdir,
                             SO3 &&R_w2l, const FresnelTerms &fres,
                             float alphax, float alphay) {
        Vec3 local_in = -R_w2l.rotate(std::forward<VType2>(indir)),
             local_out = R_w2l.rotate(std::forward<VType3>(outdir));
        // Get fresnel term
        Vec3 wh = (local_out + local_in).normalized().face_forward();
        // note that indir points inwards, hence we need negative sign to flip
        // the indir direction
        Vec4 fres_f = fres.fresnel_conductor(fabsf(local_out.dot(wh)));
        float cos_i = local_in.z(), cos_o = local_out.z();
        bool not_same_hemisphere = (cos_i > 0) ^ (cos_o > 0);
        // return D * F * G / (geo-term)
        return not_same_hemisphere
                   ? Vec4(0)
                   : (GGX::D(std::move(wh), alphax, alphay) * fres_f *
                      __frcp_rn(4.f * cos_i * cos_o) *
                      GGX::G(std::move(local_in), std::move(local_out), alphax,
                             alphay));
    }
};

CPT_CPU_GPU GGXConductorBSDF::GGXConductorBSDF(Vec3 eta_t, Vec3 k, Vec4 albedo,
                                               float roughness_x,
                                               float roughness_y)
    : BSDF(Vec4(0),
           Vec4(roughness_to_alpha(roughness_x),
                roughness_to_alpha(roughness_y), 1),
           std::move(albedo),
           ScatterStateFlag::BSDF_GLOSSY | ScatterStateFlag::BSDF_REFLECT),
      fresnel(std::move(eta_t), std::move(k)) {}

CPT_GPU float GGXConductorBSDF::pdf(const Interaction &it, const Vec3 &out,
                                    const Vec3 &in, int index) const {
    SO3 R_w2l;
    const Vec3 normal = c_textures.eval_normal_reused(it, index, R_w2l);
    const Vec3 local_in = -R_w2l.rotate(in), local_out = R_w2l.rotate(out),
               local_wh = (local_out + local_in).normalized();
    // since outdir points outward, indir points inwards, therefore the prod
    // should be negative
    const Vec2 alpha =
        c_textures.eval_rough(it.uv_coord, index, Vec2(k_s.x(), k_s.y()));
    float pdf_v = GGX::pdf(local_wh, local_in, alpha.x(), alpha.y());
    bool not_same_hemisphere = (local_in.z() > 0) ^ (local_out.z() > 0);
    return not_same_hemisphere
               ? 0
               : pdf_v * __frcp_rn(4.f * local_wh.dot(local_in));
}

CPT_GPU Vec4 GGXConductorBSDF::eval(const Interaction &it, const Vec3 &out,
                                    const Vec3 &in, int index, bool is_mi,
                                    bool is_radiance) const {
    SO3 R_w2l;
    const Vec3 normal = c_textures.eval_normal_reused(it, index, R_w2l);
    const cudaTextureObject_t glos_tex = c_textures.glos_tex[index];
    const Vec2 alpha =
        c_textures.eval_rough(it.uv_coord, index, Vec2(k_s.x(), k_s.y()));
    return c_textures.eval(glos_tex, it.uv_coord, k_g) *
           GGX::eval(normal, in, out, std::move(R_w2l), fresnel, alpha.x(),
                     alpha.y()) *
           fmaxf(0, out.dot(normal));
}

CPT_GPU Vec3 GGXConductorBSDF::sample_dir(const Vec3 &indir,
                                          const Interaction &it,
                                          Vec4 &throughput, float &pdf,
                                          Sampler &sp,
                                          ScatterStateFlag &samp_lobe,
                                          int index, bool is_radiance) const {
    SO3 R_w2l;
    const Vec3 normal = c_textures.eval_normal_reused(it, index, R_w2l);
    const Vec2 alpha =
        c_textures.eval_rough(it.uv_coord, index, Vec2(k_s.x(), k_s.y()));
    const Vec3 local_in = -R_w2l.rotate(indir), // from world to local
        local_whf = GGX::sample_wh(local_in, alpha.x(), alpha.y(), sp.next2D());

    auto D_e = GGX::D(local_whf, alpha.x(), alpha.y());
    float dot_indir_m = local_in.dot(local_whf);
    // calculate PDF
    pdf = D_e * GGX::G1(local_in, alpha.x(), alpha.y()) *
          fabsf(dot_indir_m / local_in.z());
    pdf =
        (pdf > 0 && dot_indir_m > 0) ? (pdf * __frcp_rn(4.f * dot_indir_m)) : 0;
    // calculate reflected ray direction
    Vec3 local_ref = reflection(local_in, local_whf,
                                dot_indir_m), // local space reflection vector
        refdir =
            R_w2l.transposed_rotate(local_ref); // world space reflection vector
    float cos_i = local_in.z(), cos_o = local_ref.z();
    Vec4 fres_v = fresnel.fresnel_conductor(fabsf(local_ref.dot(local_whf)));
    // calculate throughput
    if (cos_i > 0 && cos_o > 0 && pdf > 0) {
        const cudaTextureObject_t glos_tex = c_textures.glos_tex[index];
        throughput *= (1.f / pdf) *
                      c_textures.eval(glos_tex, it.uv_coord, k_g) * D_e *
                      GGX::G(local_in, local_ref, alpha.x(), alpha.y()) *
                      __frcp_rn(4.f * cos_i * cos_o) * fres_v *
                      fmaxf(normal.dot(refdir), 0);
    }
    samp_lobe = static_cast<ScatterStateFlag>(bsdf_flag);
    return refdir;
}
