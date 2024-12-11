/**
 * @file bsdf.cu
 * 
 * @author Qianyue He
 * @brief More material: rough plastic and rough cnoductor
 * @version 0.1
 * @date 2024-11-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cuda_runtime.h>
#include "core/bsdf.cuh"


/**
 * GGX microfacet model
 */
class GGX {
private:
    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static float get_lambda(VecType&& local, float alpha) {
        float e = GGX::e_func(std::forward<VecType&&>(local), alpha) * alpha * alpha;
        return e == 0 ? 0 : (-1.f + sqrtf(1.f + e)) * 0.5f;
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static float e_func(VecType&& local, float alpha) {
        float cos2_theta = local.z() * local.z(), 
              inv_cos2_theta = cos2_theta == 0 ? 0 : __frcp_rn(cos2_theta);
        // avoid calculating cos phi and sin phi
        return (local.x() * local.x() + local.y() * local.y()) * inv_cos2_theta;
    }
public:
    // input cos theta and roughness, with random sample uv, output scaled slopes
    CPT_GPU static void ggx_cos_sample(float cos_theta, float alpha, Vec2&& uv, Vec2& slopes) {
        if (cos_theta == 1) {
            float r = sqrt(uv.x() / (1 - uv.y()));
            float phi = 2.f * uv.y(), sin_phi = 0, cos_phi = 0;
            
            sincospif(phi, &sin_phi, &cos_phi);
            slopes.x() = r * cos_phi;
            slopes.y() = r * sin_phi;
            return;
        }

        float sin_theta =
            sqrtf(fmaxf(0, 1.f - cos_theta * cos_theta));
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
        slopes.x() = (A < 0 || slope_x_2 > 1.f / tan_theta) ? slope_x_1 : slope_x_2;

        // sample slope_y
        float sign = uv.y() > 0.5f ? 1.f : -1.f;
        uv.y() = 2.f * (uv.y() - .5f) * sign;

        float z =
            (uv.y() * (uv.y() * (uv.y() * 0.27385f - 0.73369f) + 0.46341f)) /
            (uv.y() * (uv.y() * (uv.y() * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
        slopes.y() = sign * z * sqrtf(1.f + slopes.x() * slopes.x());
    }

    CPT_GPU static Vec2 get_sincos_phi(const Vec3& v) {
        float sin_theta2 = fmaxf(1.f - v.z() * v.z(), 0),
              inv_sin_theta = rsqrtf(sin_theta2);
        bool theta_zero = sin_theta2 == 0;
        return Vec2(
            theta_zero ? 1 : fminf(fmaxf(v.x() * inv_sin_theta, -1), 1),
            theta_zero ? 0 : fminf(fmaxf(v.y() * inv_sin_theta, -1), 1)
        );
    }
    
    // return Vec2: (D value, e value for reuse)
    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static float D(VecType&& local, float alpha) {
        // e can be directly used to calculate G1
        float alpha2 = alpha * alpha, cos2_theta = local.z() * local.z(), 
              e = GGX::e_func(std::forward<VecType&&>(local), alpha) * __frcp_rn(alpha2);
        return __frcp_rn(M_Pi * alpha2 * cos2_theta * cos2_theta * (1 + e) * (1 + e));
    }

    CPT_GPU_INLINE static float G1_with_e(float e) {
        float lambda = e == 0 ? 0 : (-1.f + sqrtf(1.f + e)) * 0.5f;
        return 1.f / (1.f + lambda);
    }

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU_INLINE static float G(VType1&& local_in, VType2&& local_out, float alpha) {
        return 1.f / (1.f + get_lambda(local_in, alpha) + get_lambda(local_out, alpha));
    }
public:
    /**
     * @brief sample microfacet normal
     * the inputs should be in the local frame
     */
    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static Vec3 sample_wh(VecType&& local_indir, float alpha, Vec2&& uv) {
        Vec3 wi_stretched = Vec3(local_indir.x() * alpha, local_indir.y() * alpha, local_indir.z()).normalized();
        Vec2 slope;

        ggx_cos_sample(wi_stretched.z(), alpha, std::move(uv), slope);
        Vec2 cos_sin_phi = get_sincos_phi(wi_stretched);

        float tmp = cos_sin_phi.x() * slope.x() - cos_sin_phi.y() * slope.y();
        slope.y() = (cos_sin_phi.y() * slope.x() + cos_sin_phi.x() * slope.y()) * alpha;
        slope.x() = tmp * alpha;

        return Vec3(-slope.x(), -slope.y(), 1.).normalized();
    }

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU_INLINE static float pdf(VType1&& local_wh, VType2&& local_in, float alpha) {
        auto D_e = D(local_wh, alpha);
        // can be 0 for the denominator
        float cos_ratio = fabsf(local_in.dot(local_wh)) / fabsf(local_in.z());
        return D_e * G1(std::forward<VType2&&>(local_in), alpha) * cos_ratio;
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static float G1(VecType&& local, float alpha) {
        float cos2_theta = local.z() * local.z(), 
              inv_cos2_theta = cos2_theta == 0 ? 0 : __frcp_rn(cos2_theta), alpha2 = alpha * alpha;
        // avoid calculating cos phi and sin phi
        float e = (local.x() * local.x() + local.y() * local.y()) * inv_cos2_theta * alpha2;
        return G1_with_e(e);
    }

    // indir points towards the incident point, outdir points away from it
    CONDITION_TEMPLATE_SEP_3(VType1, VType2, VType3, Vec3, Vec3, Vec3)
    CPT_GPU static Vec4 eval(VType1&& n_s, VType2&& indir, VType3&& outdir, const FresnelTerms& fres, float alpha) {
        auto R_w2l = rotation_fixed_anchor(std::forward<VType1&&>(n_s), false);
        Vec3 local_in  = -R_w2l.rotate(std::forward<VType2&&>(indir)),
             local_out = R_w2l.rotate(std::forward<VType3&&>(outdir));
        // Get fresnel term
        Vec3 wh = (local_out + local_in).normalized().face_forward();
        // note that indir points inwards, hence we need negative sign to flip the indir direction
        Vec4 fres_f = fres.fresnel_conductor(fabsf(local_out.dot(wh)));
        float cos_i = local_in.z(), cos_o = local_out.z();
        bool not_same_hemisphere = (cos_i > 0) ^ (cos_o > 0);
        // return D * F * G / (geo-term)
        return not_same_hemisphere ? Vec4(0) : 
                (GGX::D(std::move(wh), alpha) * 
                fres_f * __frcp_rn(4.f * cos_i * cos_o) *
                GGX::G(std::move(local_in), std::move(local_out), alpha));
    }
};

CPT_CPU_GPU PlasticBSDF::PlasticBSDF(
    Vec4 _k_d, Vec4 _k_s, Vec4 sigma_a, float ior, 
    float trans_scaler, float thickness, int kd_id, int ks_id
): BSDF(_k_d, std::move(_k_s), std::move(sigma_a), kd_id, ks_id, 
    BSDFFlag::BSDF_SPECULAR | 
    BSDFFlag::BSDF_DIFFUSE  | 
    BSDFFlag::BSDF_REFLECT
), trans_scaler(trans_scaler), thickness(thickness), eta(1.f / ior) {
    precomp_diff_f = FresnelTerms::diffuse_fresnel(ior);
}

CPT_GPU float PlasticBSDF::pdf(const Interaction& it, const Vec3& out, const Vec3& in) const {
    float dot_wo = out.dot(it.shading_norm), dot_wi = in.dot(it.shading_norm),
          Fi = FresnelTerms::fresnel_simple(eta, -dot_wi),
          specular_prob = Fi / (Fi + trans_scaler * (1.0f - Fi));
    Vec3 refdir = -reflection(in, it.shading_norm);
    // 'refdir.dot(out) >= 1.f - THP_EPS' means reflected incident ray is very close to 'out'
    float pdf = refdir.dot(out) < 1.f - THP_EPS ? M_1_Pi * dot_wo * (1.f - specular_prob) : specular_prob;
    return dot_wo > 0 && dot_wi < 0 ? pdf : 0;
}

CPT_GPU Vec4 PlasticBSDF::eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi, bool is_radiance) const {
    float dot_wo = out.dot(it.shading_norm), dot_wi = in.dot(it.shading_norm),
          Fi = FresnelTerms::fresnel_simple(eta, fabsf(dot_wi)),
          Fo = FresnelTerms::fresnel_simple(eta, dot_wo);
    Vec3 refdir = -reflection(in, it.shading_norm);

    Vec4 brdf = (M_1_Pi * (1.0f - Fi) * (1.0f - Fo) * eta * eta) * dot_wo *
            (k_d / (- k_d * precomp_diff_f + 1.f)) * 
            (k_g * thickness * (-1.0f / dot_wo + 1.0f / dot_wi)).exp_xyz();
    brdf += refdir.dot(out) < 1.f - THP_EPS ? brdf : Vec4(Fi, 1);

    return dot_wo > 0 && dot_wi < 0 ? brdf : Vec4(0, 1);
}

CPT_GPU Vec3 PlasticBSDF::sample_dir(
    const Vec3& indir, 
    const Interaction& it, 
    Vec4& throughput, 
    float& pdf, Sampler& sp, 
    BSDFFlag& samp_lobe, 
    bool is_radiance
) const {
    float dot_indir = it.shading_norm.dot(indir);
    throughput = dot_indir < 0 ? throughput : Vec4(0, 1);

    float Fi = FresnelTerms::fresnel_simple(eta, fabsf(dot_indir)),
          specular_prob = Fi / (Fi + trans_scaler * (1.0f - Fi));

    Vec3 outdir;

    if (sp.next1D() < specular_prob) {      // coating specular reflection
        outdir = -reflection(indir, it.shading_norm, dot_indir);
        pdf = specular_prob;
        throughput *= Vec4(Fi / specular_prob, 1) * k_s;
        samp_lobe = static_cast<BSDFFlag>(BSDFFlag::BSDF_REFLECT | BSDFFlag::BSDF_SPECULAR);
    } else {                                // substrate diffuse reflection
        float dummy_v = 1;
        Vec3 local_dir = sample_cosine_hemisphere(sp.next2D(), dummy_v);
        float Fo = FresnelTerms::fresnel_simple(eta, local_dir.z());
       
        // use fmaxf instead of (1-Fi) * (1-Fo), to make the result looks brighter
        Vec4 local_thp = ((1.0f - Fi) * (1.0f - Fo) * eta * eta) * (k_d / (-k_d * precomp_diff_f + 1.f)) * 
            (k_g * thickness * (-1.0f / local_dir.z() + 1.0f / dot_indir)).exp_xyz();

        pdf = M_1_Pi * local_dir.z() * (1.0f - specular_prob);
        throughput *=  __frcp_rn(1.f - specular_prob) * local_thp;
        outdir = delocalize_rotate(it.shading_norm, local_dir);
        samp_lobe = static_cast<BSDFFlag>(BSDFFlag::BSDF_REFLECT | BSDFFlag::BSDF_DIFFUSE);
    }
    return outdir;
}

CPT_GPU float GGXMetalBSDF::pdf(const Interaction& it, const Vec3& out, const Vec3& in) const {
    auto R_w2l = rotation_fixed_anchor(it.shading_norm, false);
    const Vec3 local_in  = -R_w2l.rotate(in),
               local_out = R_w2l.rotate(out),
               local_wh  = (local_out + local_in).normalized();
    // since outdir points outward, indir points inwards, therefore the prod should be negative
    float pdf_v = GGX::pdf(local_wh, local_in, k_s.w());
    bool not_same_hemisphere = (local_in.z() > 0) ^ (local_out.z() > 0);
    return not_same_hemisphere ? 0 : pdf_v * __frcp_rn(4.f * local_wh.dot(local_in));
}

CPT_GPU Vec4 GGXMetalBSDF::eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi, bool is_radiance) const {
    return k_g * GGX::eval(it.shading_norm, in, out, fresnel, k_s.w()) * fmaxf(0, out.dot(it.shading_norm));
}

CPT_GPU Vec3 GGXMetalBSDF::sample_dir(    
    const Vec3& indir, 
    const Interaction& it, 
    Vec4& throughput, 
    float& pdf, Sampler& sp, 
    BSDFFlag& samp_lobe, 
    bool is_radiance
) const {
    auto R_w2l     = rotation_fixed_anchor(it.shading_norm, false);         // transpose will be l2w
    const Vec3 local_in = -R_w2l.rotate(indir),                             // from world to local
          local_whf     = GGX::sample_wh(local_in, k_s.w(), sp.next2D());

    auto D_e = GGX::D(local_whf, k_s.w());
    float dot_indir_m = local_in.dot(local_whf);
    // calculate PDF
    pdf = D_e * GGX::G1(local_in, k_s.w()) * fabsf(dot_indir_m / local_in.z());
    pdf = (pdf > 0 && dot_indir_m > 0) ? (pdf * __frcp_rn(4.f * dot_indir_m)) : 0;
    // calculate reflected ray direction
    Vec3 local_ref = reflection(local_in, local_whf, dot_indir_m),      // local space reflection vector
         refdir    = R_w2l.transposed_rotate(local_ref);                // world space reflection vector
    float cos_i = local_in.z(), cos_o = local_ref.z();
    Vec4 fres_v = fresnel.fresnel_conductor(fabsf(local_ref.dot(local_whf)));
    // calculate throughput
    if (cos_i > 0 && cos_o > 0 && pdf > 0) {
        throughput *= (1.f / pdf) * k_g * D_e * GGX::G(local_in, local_ref, k_s.w()) * 
        __frcp_rn(4.f * cos_i * cos_o) * fres_v * fmaxf(it.shading_norm.dot(refdir), 0);
    }
    samp_lobe = static_cast<BSDFFlag>(bsdf_flag);
    return refdir;                              
}
