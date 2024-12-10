/**
 * CUDA object information
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/vec2.cuh"
#include "core/vec4.cuh"
#include "core/sampling.cuh"
#include "core/interaction.cuh"
#include "core/metal_params.cuh"

enum BSDFFlag: int {
    BSDF_NONE     = 0x00,
    BSDF_DIFFUSE  = 0x01,
    BSDF_SPECULAR = 0x02,
    BSDF_GLOSSY   = 0x04,
    BSDF_FORWARD  = 0x08,

    BSDF_REFLECT  = 0x10,
    BSDF_TRANSMIT = 0x20
};

class BSDF {
public:
    Vec4 k_d;
    Vec4 k_s;
    Vec4 k_g;
    int kd_tex_id;
    int ex_tex_id;
    int bsdf_flag;
    int __padding;
public:
    CPT_CPU_GPU BSDF() {}
    CPT_CPU_GPU BSDF(Vec4 _k_d, Vec4 _k_s, Vec4 _k_g, int kd_id = -1, int kg_id = -1, int flag = BSDFFlag::BSDF_NONE):
        k_d(std::move(k_d)), k_s(std::move(_k_s)), k_g(std::move(_k_g)),
        kd_tex_id(kd_id), ex_tex_id(kg_id), bsdf_flag(flag)
    {}

    CPT_GPU void set_kd(Vec4&& v) noexcept { this->k_d = v; }
    CPT_GPU void set_ks(Vec4&& v) noexcept { this->k_s = v; }
    CPT_GPU void set_kg(Vec4&& v) noexcept { this->k_g = v; }
    CPT_GPU void set_kd_id(int v) noexcept { this->kd_tex_id = v; }
    CPT_GPU void set_ex_id(int v) noexcept { this->ex_tex_id = v; }
    CPT_GPU void set_lobe(int v) noexcept { this->bsdf_flag = v; }

    CPT_GPU virtual float pdf(const Interaction& it, const Vec3& out, const Vec3& in) const = 0;

    CPT_GPU virtual Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false, bool is_radiance = true) const = 0;

    CPT_GPU virtual Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float&, Sampler& sp, bool is_radiance = true) const = 0;

    CPT_GPU_INLINE bool require_lobe(BSDFFlag flags) const noexcept {
        return (bsdf_flag & (int)flags) > 0;
    }

    static CPT_CPU_GPU_INLINE float roughness_to_alpha(float roughness) {
        roughness = fmaxf(roughness, 1e-3f);
        float x = logf(roughness);
        return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
           0.000640711f * x * x * x * x;
    }
};


class LambertianBSDF: public BSDF {
using BSDF::k_d;
using BSDF::bsdf_flag;
public:
    CPT_CPU_GPU LambertianBSDF(Vec4 _k_d, int kd_id = -1):
        BSDF(std::move(_k_d), Vec4(0, 0, 0), Vec4(0, 0, 0), kd_id, -1, BSDFFlag::BSDF_DIFFUSE | BSDFFlag::BSDF_REFLECT) {}

    CPT_CPU_GPU LambertianBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override {
        // printf("it.norm: %f, %f, %f\n", it.shading_norm.x(), it.shading_norm.y(), it.shading_norm.z());
        // printf("out: %f, %f, %f\n", out.x(), out.y(), out.z());
        float dot_val = it.shading_norm.dot(out);
        return max(it.shading_norm.dot(out), 0.f) * M_1_Pi;
    }

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false, bool is_radiance = true) const override {
        float cos_term = it.shading_norm.dot(out);
        float dot_in  = it.shading_norm.dot(in);
        float same_side = (dot_in > 0) ^ (cos_term > 0);     // should be positive or negative at the same time
        // printf("%f, k_d: %f, %f, %f\n", cosine_term, k_d.x(), k_d.y(), k_d.z());
        return k_d * max(0.f, cos_term) * M_1_Pi * same_side;
    }

    CPT_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, Sampler& sp, bool is_radiance = true) const override {
        auto local_ray = sample_cosine_hemisphere(sp.next2D(), pdf);
        auto out_ray = delocalize_rotate(it.shading_norm, local_ray);
        // throughput *= f / pdf --> k_d * cos / pi / (pdf = cos / pi) == k_d
        float dot_in  = it.shading_norm.dot(indir);
        float dot_out = it.shading_norm.dot(out_ray);
        throughput *= k_d * ((dot_in > 0) ^ (dot_out > 0));
        return out_ray;
    }
};

class SpecularBSDF: public BSDF {
using BSDF::k_s;
public:
    CPT_CPU_GPU SpecularBSDF(Vec4 _k_s, int ks_id = -1):
        BSDF(Vec4(0, 0, 0), std::move(_k_s), Vec4(0, 0, 0), -1, ks_id, BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_REFLECT) {}

    CPT_CPU_GPU SpecularBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override {
        return 0.f;
    }

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false, bool is_radiance = true) const override {
        auto ref_dir = in.advance(it.shading_norm, -2.f * in.dot(it.shading_norm)).normalized();
        return k_s * (out.dot(ref_dir) > 0.99999f);
    }

    CPT_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, Sampler& sp, bool is_radiance = true) const override {
        // throughput *= f / pdf
        float in_dot_n = indir.dot(it.shading_norm);
        pdf = 1.f;
        throughput *= k_s * (in_dot_n < 0);
        return reflection(indir, it.shading_norm, in_dot_n);
    }
};

class FresnelTerms {
private:
    Vec3 eta_t;     // for conductor
    Vec3 k;         // for conductor
public:
    CPT_CPU_GPU FresnelTerms() {}

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU FresnelTerms(VType1&& _eta_t, VType2&& _k): 
        eta_t(std::forward<VType1&&>(_eta_t)),
        k(std::forward<VType1&&>(_k)) {}

    CPT_GPU_INLINE static bool is_total_reflection(float dot_normal, float ni, float nr) {
        return (1.f - (ni * ni) / (nr * nr) * (1.f - dot_normal * dot_normal)) < 0.f;
    }

    CPT_GPU static Vec3 snell_refraction(const Vec3& incid, const Vec3& normal, float& cos_r2, float dot_n, float ni, float nr) {
        /* Refraction vector by Snell's Law, note that an extra flag will be returned */
        float ratio = ni / nr;
        cos_r2 = 1.f - (ratio * ratio) * (1. - dot_n * dot_n);        // refraction angle cosine
        // for ni > nr situation, there will be total reflection
        // if cos_r2 <= 0.f, then return value will be Vec3(0, 0, 0)
        return (ratio * incid - ratio * dot_n * normal + sgn(dot_n) * sqrtf(fabsf(cos_r2)) * normal).normalized() * (cos_r2 > 0.f);
    }

    CPT_GPU static float fresnel_dielectric(float n_in, float n_out, float cos_inc, float cos_ref) {
        /**
            Fresnel Equation for calculating specular ratio
            Since Schlick's Approximation is not clear about n1->n2, n2->n1 (different) effects

            This Fresnel equation is for dielectric, not for conductor
        */
        float n1cos_i = n_in * cos_inc;
        float n2cos_i = n_out * cos_inc;
        float n1cos_r = n_in * cos_ref;
        float n2cos_r = n_out * cos_ref;
        float rs = (n1cos_i - n2cos_r) / (n1cos_i + n2cos_r);
        float rp = (n1cos_r - n2cos_i) / (n1cos_r + n2cos_i);
        return 0.5f * (rs * rs + rp * rp);
    }

    CPT_GPU Vec4 fresnel_conductor(float cos_theta_i) const {
        cos_theta_i = fminf(fmaxf(cos_theta_i, -1), 1);

        float cos2_theta_i = cos_theta_i * cos_theta_i;
        float sin2_theta_i = 1. - cos2_theta_i;
        Vec3 eta2  = eta_t * eta_t;
        Vec3 etak2 = k * k;

        Vec3 t0 = eta2 - etak2 - sin2_theta_i;
        Vec3 a2plusb2 = t0 * t0 + 4 * eta2 * etak2;
        a2plusb2 = Vec3(sqrtf(a2plusb2.x()), sqrtf(a2plusb2.y()), sqrtf(a2plusb2.z()));
        Vec3 t1 = a2plusb2 + cos2_theta_i;
        Vec3 a = 0.5f * (a2plusb2 + t0);
        a = Vec3(sqrtf(a.x()), sqrtf(a.y()), sqrtf(a.z()));

        Vec3 t2 = 2.f * cos_theta_i * a;
        Vec3 Rs = (t1 - t2) / (t1 + t2);

        Vec3 t3 = cos2_theta_i * a2plusb2 + sin2_theta_i * sin2_theta_i;
        Vec3 t4 = t2 * sin2_theta_i;
        Vec3 Rp = Rs * (t3 - t4) / (t3 + t4);

        Rp = 0.5f * (Rp + Rs);
        return Vec4(Rp.x(), Rp.y(), Rp.z(), 1);
    }
};


class TranslucentBSDF: public BSDF {
using BSDF::k_s;        // specular reflection
using BSDF::k_d;        // ior
public:
    CPT_CPU_GPU TranslucentBSDF(Vec4 k_s, Vec4 ior, int ex_id):
        BSDF(std::move(ior), std::move(k_s), Vec4(0, 0, 0), -1, ex_id, BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_TRANSMIT) {}

    CPT_CPU_GPU TranslucentBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& incid) const override {
        return 0.f;
    }

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false, bool is_radiance = true) const override {
        float dot_normal = in.dot(it.shading_norm);
        // at least according to pbrt-v3, ni / nr is computed as the following (using shading normal)
        // see https://computergraphics.stackexchange.com/questions/13540/shading-normal-and-geometric-normal-for-refractive-surface-rendering
        float ni = dot_normal < 0 ? 1.f : k_d.x();
        float nr = dot_normal < 0 ? k_d.x() : 1.f, cos_r2 = 0;
        float eta2 = ni * ni / (nr * nr);
        Vec3 ret_dir = in.advance(it.shading_norm, -2.f * dot_normal).normalized(),
             refra_vec = FresnelTerms::snell_refraction(in, it.shading_norm, cos_r2, dot_normal, ni, nr);
        bool total_ref = FresnelTerms::is_total_reflection(dot_normal, ni, nr);
        nr = FresnelTerms::fresnel_dielectric(ni, nr, fabsf(dot_normal), sqrtf(fabsf(cos_r2)));
        bool reflc_dot = out.dot(ret_dir) > 0.99999f, refra_dot = out.dot(refra_vec) > 0.99999f;        // 0.9999  means 0.26 deg
        
        return k_s * (reflc_dot | refra_dot) * (refra_dot && is_radiance ? eta2 : 1.f);
    }

    CPT_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, Sampler& sp, bool is_radiance = true) const override {
        float dot_normal = indir.dot(it.shading_norm);
        // at least according to pbrt-v3, ni / nr is computed as the following (using shading normal)
        // see https://computergraphics.stackexchange.com/questions/13540/shading-normal-and-geometric-normal-for-refractive-surface-rendering
        float ni = dot_normal < 0 ? 1.f : k_d.x();
        float nr = dot_normal < 0 ? k_d.x() : 1.f, cos_r2 = 0;
        float eta2 = ni * ni / (nr * nr);
        Vec3 ret_dir = indir.advance(it.shading_norm, -2.f * dot_normal).normalized(),
             refra_vec = FresnelTerms::snell_refraction(indir, it.shading_norm, cos_r2, dot_normal, ni, nr);
        bool total_ref = FresnelTerms::is_total_reflection(dot_normal, ni, nr);
        nr = FresnelTerms::fresnel_dielectric(ni, nr, fabsf(dot_normal), sqrtf(fabsf(cos_r2)));
        bool reflect = total_ref || sp.next1D() < nr;
        ret_dir = select(ret_dir, refra_vec, reflect);
        pdf     = total_ref ? 1.f : (reflect ? nr : 1.f - nr);
        throughput *= k_s * (is_radiance && !reflect ? eta2 : 1.f);
        return ret_dir;
    }
};

class RoughPlasticBSDF: public BSDF {
using BSDF::k_s;
public:
    CPT_CPU_GPU RoughPlasticBSDF(Vec4 _k_d, Vec4 _k_s, float ior, float roughness, int ks_id = -1):
        BSDF(std::move(_k_d), std::move(_k_s), Vec4(roughness_to_alpha(roughness), ior, 0), -1, ks_id, 
            BSDFFlag::BSDF_GLOSSY  | 
            BSDFFlag::BSDF_DIFFUSE | 
            BSDFFlag::BSDF_REFLECT
        ) {}

    CPT_CPU_GPU RoughPlasticBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override;

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false, bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, Sampler& sp, bool is_radiance = true) const override;
};

class GGXMetalBSDF: public BSDF {
/**
 * @brief GGX microfacet normal distribution based BSDF
 * k_d is the eta_t of the metal
 * k_s is the k (Vec3) and the mapped roughness (k_s[3])
 * k_g is the underlying color (albedo)
 */
using BSDF::k_s;
private:
    const FresnelTerms fresnel;
public:
    CPT_CPU_GPU GGXMetalBSDF(Vec3 eta_t, Vec3 k, Vec4 albedo, float roughness, int ks_id = -1):
        BSDF(Vec4(0), Vec4(roughness_to_alpha(roughness)), 
            std::move(albedo), -1, ks_id, BSDFFlag::BSDF_GLOSSY | BSDFFlag::BSDF_REFLECT), 
            fresnel(std::move(eta_t), std::move(k)) {}

    CPT_CPU_GPU GGXMetalBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override;

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false, bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, Sampler& sp, bool is_radiance = true) const override;
};