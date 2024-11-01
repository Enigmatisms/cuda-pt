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

    CPT_GPU virtual Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float&, Vec2&& uv, bool is_radiance = true) const = 0;

    CPT_GPU_INLINE bool require_lobe(BSDFFlag flags) const noexcept {
        return (bsdf_flag & (int)flags) > 0;
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

    CPT_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, Vec2&& uv, bool is_radiance = true) const override {
        auto local_ray = sample_cosine_hemisphere(std::move(uv), pdf);
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

    CPT_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, Vec2&& uv, bool is_radiance = true) const override {
        // throughput *= f / pdf
        pdf = 1.f;
        throughput *= k_s * (indir.dot(it.shading_norm) < 0);
        return indir - 2.f * indir.dot(it.shading_norm) * it.shading_norm;
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
             refra_vec = snell_refraction(in, it.shading_norm, cos_r2, dot_normal, ni, nr);
        bool total_ref = is_total_reflection(dot_normal, ni, nr);
        nr = fresnel_equation(ni, nr, fabsf(dot_normal), sqrtf(fabsf(cos_r2)));
        bool reflc_dot = out.dot(ret_dir) > 0.99999f, refra_dot = out.dot(refra_vec) > 0.99999f;        // 0.9999  means 0.26 deg
        
        return k_s * (reflc_dot | refra_dot) * (refra_dot && is_radiance ? eta2 : 1.f);
    }

    CPT_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, Vec2&& uv, bool is_radiance = true) const override {
        float dot_normal = indir.dot(it.shading_norm);
        // at least according to pbrt-v3, ni / nr is computed as the following (using shading normal)
        // see https://computergraphics.stackexchange.com/questions/13540/shading-normal-and-geometric-normal-for-refractive-surface-rendering
        float ni = dot_normal < 0 ? 1.f : k_d.x();
        float nr = dot_normal < 0 ? k_d.x() : 1.f, cos_r2 = 0;
        float eta2 = ni * ni / (nr * nr);
        Vec3 ret_dir = indir.advance(it.shading_norm, -2.f * dot_normal).normalized(),
             refra_vec = snell_refraction(indir, it.shading_norm, cos_r2, dot_normal, ni, nr);
        bool total_ref = is_total_reflection(dot_normal, ni, nr);
        nr = fresnel_equation(ni, nr, fabsf(dot_normal), sqrtf(fabsf(cos_r2)));
        bool reflect = total_ref || uv.x() < nr;
        ret_dir = select(ret_dir, refra_vec, reflect);
        pdf     = total_ref ? 1.f : (reflect ? nr : 1.f - nr);
        // throughput *= f / pdf
        throughput *= k_s * (is_radiance && !reflect ? eta2 : 1.f);
        return ret_dir;
    }

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

    CPT_GPU static float fresnel_equation(float n_in, float n_out, float cos_inc, float cos_ref) {
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
};