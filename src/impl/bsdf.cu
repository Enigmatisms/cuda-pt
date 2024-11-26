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
public:
    // input cos theta and roughness, with random sample uv, output scaled slopes
    CPT_GPU static void ggx_cos_sample(float cos_theta, float alpha, Vec2&& uv, Vec2& slopes) {
        if (cos_theta > 0.9999f) {
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
            theta_zero ? 1 : fminf(fmaxf(v.y() * inv_sin_theta, -1), 1)
        );
    }

    CPT_GPU static float get_sin_phi(const Vec3& v) {
        float sin_theta = sqrtf(fmaxf(1.f - v.z() * v.z(), 0));
        return (sin_theta == 0) ? 1 : fminf(fmaxf(v.x() / sin_theta, -1), 1);
    }

    // sample the local half vector
    CPT_GPU static Vec3 sample_local(const Vec3& local_indir, float alpha, Vec2&& uv) {
        Vec3 wi_stretched = Vec3(local_indir.x() * alpha, local_indir.y() * alpha, local_indir.z()).normalized();
        Vec2 slope;

        ggx_cos_sample(wi_stretched.z(), alpha, std::move(uv), slope);
        Vec2 cos_sin_phi = get_sincos_phi(wi_stretched);

        float tmp = cos_sin_phi.x() * slope.x() - cos_sin_phi.y() * slope.y();
        slope.y() = (cos_sin_phi.y() * slope.x() + cos_sin_phi.x() * slope.y()) * alpha;
        slope.x() = tmp * alpha;

        return Vec3(-slope.x(), -slope.y(), 1.).normalized();
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static float e_func(VecType&& local, float alpha) {
        float cos2_theta = local.z() * local.z(), 
              inv_cos2_theta = cos2_theta == 0 ? 0 : __frcp_rn(cos2_theta), alpha2 = alpha * alpha;
        // avoid calculating cos phi and sin phi
        return (local.x() * local.x() + local.y() * local.y()) * inv_cos2_theta * __frcp_rn(alpha2);
    }
    
    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static Vec2 D(VecType&& local, float alpha) {
        float e = GGX::e_func(std::forward<VecType&&>(local), alpha);
        // e can be directly used to calculate G1
        return Vec2(__frcp_rn(M_Pi * alpha2 * cos2_theta * cos2_theta * (1 + e) * (1 + e)), e);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static float get_lambda(VecType&& local, float alpha) {
        float e = GGX::e_func(std::forward<VecType&&>(local), alpha);
        return e == 0 ? 0 : (-1.f + sqrtf(1.f + e)) * 0.5f;
    }

    CPT_GPU_INLINE static float G1_with_e(float e) {
        float lambda = e == 0 ? 0 : (-1.f + sqrtf(1.f + e)) * 0.5f;
        return 1.f / (1.f + lambda);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static float G1(VecType&& local, float alpha) {
        float cos2_theta = local.z() * local.z(), 
              inv_cos2_theta = cos2_theta == 0 ? 0 : __frcp_rn(cos2_theta), alpha2 = alpha * alpha;
        // avoid calculating cos phi and sin phi
        float e = (local.x() * local.x() + local.y() * local.y()) * inv_cos2_theta * __frcp_rn(alpha2);
        return G1_with_e(e);
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

    CONDITION_TEMPLATE_SEP_3(VType1, VType2, VType3, Vec3, Vec3, Vec3)
    CPT_CPU_GPU static float pdf(VType1&& n_s, VType2&& wh, VType3&& indir, float alpha) {
        auto R_w2l = rotation_fixed_anchor(std::forward<VType1&&>(n_s), false);
        Vec3 local_wh = R_w2l.rotate(std::forward<VType2&&>(wh)),
             local_in = R_w2l.rotate(std::forward<VType3&&>(indir));
        auto D_e = D(local_wh, alpha);
        // can be 0 for the denominator
        return D_e.x() * G1_with_e(D_e.y()) * fabsf(indir.dot(wh)) / fabsf(local_in.z());
    }

    CONDITION_TEMPLATE_SEP_3(VType1, VType2, VType3, Vec3, Vec3, Vec3)
    CPT_CPU_GPU static float eval(VType1&& n_s, VType2&& indir, VType3&& outdir, float alpha) {
        auto R_w2l = rotation_fixed_anchor(std::forward<VType1&&>(n_s), false);
        Vec3 local_in  = R_w2l.rotate(std::forward<VType2&&>(indir)),
             local_out = R_w2l.rotate(std::forward<VType3&&>(outdir));
        // Get fresnel term
    }
};

CPT_GPU float RoughPlasticBSDF::pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const {
    return 0;
}

CPT_GPU Vec4 RoughPlasticBSDF::eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi, bool is_radiance) const {
    return Vec4();
}

CPT_GPU Vec3 RoughPlasticBSDF::sample_dir(
    const Vec3& indir, 
    const Interaction& it, 
    Vec4& throughput, 
    float& pdf, Sampler& sp, 
    bool is_radiance
) const {
    return Vec3();
}

CPT_GPU float GGXMetalBSDF::pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const {
    return 0;
}

CPT_GPU Vec4 GGXMetalBSDF::eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi, bool is_radiance) const {
    return Vec4();
}

CPT_GPU Vec3 GGXMetalBSDF::sample_dir(    
    const Vec3& indir, 
    const Interaction& it, 
    Vec4& throughput, 
    float& pdf, Sampler& sp, 
    bool is_radiance
) const {
    auto R_w2l     = rotation_fixed_anchor(it.shading_norm, false);     // transpose will be l2w
    Vec3 local_in = R_w2l.rotate(indir), m_normal,                   // from world to local
         local_whf = GGX::sample_wh(local_in, k_s.w(), sp.next2D());

    auto D_e = GGX::D(local_whf, k_s.w());
    float dot_indir_m = local_in.dot(local_whf);
    // calculate PDF
    pdf = D_e.x() * GGX::G1_with_e(D_e.y()) * fabsf(dot_indir_m) / fabsf(local_in.z());
    pdf = pdf > 0 && dot_indir_m > 1e-5f ? pdf / (4.f * dot_indir_m) : 0;
    // calculate reflected ray direction
    Vec3 local_ref = reflection(local_in, local_whf, dot_indir_m),            // local space reflection vector
         refdir = R_w2l.transposed_rotate(local_ref);                         // world space reflection vector
    float cos_i = fabsf(local_in.z()), cos_o = fabsf(local_ref.z());
    bool valid_result = cos_i > 0 && cos_o > 0;
    // calculate throughput
    throughput *= valid_result ? k_g * D_e.x() * GGX::G(local_in, local_ref, k_s.w()) * __frcp_rn(4.f * cos_i * cos_o) *
        FresnelTerms::fresnel_conductor(fabsf(dot_indir_m), Vec3(k_d.xyz()), Vec3(k_s.xyz())) : Vec4(0);
    return refdir;
}