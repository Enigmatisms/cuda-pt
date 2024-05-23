/**
 * CUDA object information
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/vec2.cuh"
#include "core/sampling.cuh"
#include "core/interaction.cuh"

class BSDF {
public:
    Vec3 k_d;
    Vec3 k_s;
    Vec3 k_g;
    int kd_tex_id;
    int ex_tex_id;
public:
    CPT_CPU_GPU BSDF() {}
    CPT_CPU_GPU BSDF(Vec3 _k_d, Vec3 _k_s, Vec3 _k_g, int kd_id = -1, int kg_id = -1):
        k_d(std::move(k_d)), k_s(std::move(_k_s)), k_g(std::move(_k_g)),
        kd_tex_id(kd_id), ex_tex_id(kg_id)
    {}

    CPT_GPU void set_kd(Vec3&& v) noexcept { this->k_d = v; }
    CPT_GPU void set_ks(Vec3&& v) noexcept { this->k_s = v; }
    CPT_GPU void set_kg(Vec3&& v) noexcept { this->k_g = v; }
    CPT_GPU void set_kd_id(int v) noexcept { this->kd_tex_id = v; }
    CPT_GPU void set_ex_id(int v) noexcept { this->ex_tex_id = v; }

    CPT_CPU_GPU virtual float pdf(const Interaction& it, const Vec3& out, const Vec3& in) const = 0;

    CPT_CPU_GPU virtual Vec3 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false) const = 0;

    CPT_CPU_GPU virtual Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec3& throughput, Vec2&& uv) const = 0;
};


class LambertianBSDF: public BSDF {
using BSDF::k_d;
public:
    CPT_CPU_GPU LambertianBSDF(Vec3 _k_d, int kd_id = -1):
        BSDF(std::move(_k_d), Vec3(0, 0, 0), Vec3(0, 0, 0), kd_id, -1) {}

    CPT_CPU_GPU LambertianBSDF(): BSDF() {}
    
    CPT_CPU_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override {
        // printf("it.norm: %f, %f, %f\n", it.shading_norm.x(), it.shading_norm.y(), it.shading_norm.z());
        // printf("out: %f, %f, %f\n", out.x(), out.y(), out.z());
        float dot_val = it.shading_norm.dot(out);
        return max(it.shading_norm.dot(out), 0.f) * M_1_PI;
    }

    CPT_CPU_GPU Vec3 eval(const Interaction& it, const Vec3& out, const Vec3& /*in */, bool is_mi = false) const override {
        float cosine_term = it.shading_norm.dot(out);
        // printf("%f, k_d: %f, %f, %f\n", cosine_term, k_d.x(), k_d.y(), k_d.z());
        return k_d * max(0.f, cosine_term) * M_1_PI;
    }

    CPT_CPU_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec3& throughput, Vec2&& uv) const override {
        float sample_pdf = 0;
        auto local_ray = sample_cosine_hemisphere(std::move(uv), sample_pdf);
        // throughput *= f / pdf
        throughput *= k_d;
        return delocalize_rotate(Vec3(0, 0, 1), it.shading_norm, local_ray);;
    }
};

class SpecularBSDF: public BSDF {
using BSDF::k_s;
public:
    CPT_CPU_GPU SpecularBSDF(Vec3 _k_s, int ks_id = -1):
        BSDF(Vec3(0, 0, 0), std::move(_k_s), Vec3(0, 0, 0), -1, ks_id) {}

    CPT_CPU_GPU SpecularBSDF(): BSDF() {}
    
    CPT_CPU_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override {
        return 0.f;
    }

    CPT_CPU_GPU Vec3 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false) const override {
        return Vec3(0, 0, 0);
    }

    CPT_CPU_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec3& throughput, Vec2&& uv) const override {
        // throughput *= f / pdf
        throughput *= k_s * (indir.dot(it.shading_norm) < 0);
        return indir - 2.f * indir.dot(it.shading_norm) * it.shading_norm;
    }
};


class TranslucentBSDF: public BSDF {
using BSDF::k_s;        // specular reflection
using BSDF::k_d;        // ior
public:
    CPT_CPU_GPU TranslucentBSDF(Vec3 k_s, Vec3 ior, int ex_id):
        BSDF(std::move(ior), std::move(k_s), Vec3(0, 0, 0), -1, ex_id) {}

    CPT_CPU_GPU TranslucentBSDF(): BSDF() {}
    
    CPT_CPU_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& incid) const override {
        return 0.f;
    }

    CPT_CPU_GPU Vec3 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false) const override {
        return Vec3(0, 0, 0);
    }

    CPT_CPU_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec3& throughput, Vec2&& uv) const override {
        float dot_normal = indir.dot(it.shading_norm);
        // at least according to pbrt-v3, ni / nr is computed as the following (using shading normal)
        // see https://computergraphics.stackexchange.com/questions/13540/shading-normal-and-geometric-normal-for-refractive-surface-rendering
        float ni = select(1.f, k_d.x(), dot_normal < 0);
        float nr = select(k_d.x(), 1.f, dot_normal < 0), cos_r2 = 0;
        Vec3 ret_dir = (indir - 2.f * it.shading_norm * dot_normal).normalized(),
             refra_vec = snell_refraction(indir, it.shading_norm, cos_r2, dot_normal, ni, nr);
        bool total_ref = is_total_reflection(dot_normal, ni, nr), reflect = uv.x() < \
            fresnel_equation(ni, nr, fabsf(dot_normal), sqrtf(fabsf(cos_r2)));
        ret_dir = select(ret_dir, refra_vec, total_ref || reflect);
        // throughput *= f / pdf
        throughput *= k_s;
        return ret_dir;
    }

    CPT_CPU_GPU_INLINE static bool is_total_reflection(float dot_normal, float ni, float nr) {
        return (1.f - (ni * ni) / (nr * nr) * (1.f - dot_normal * dot_normal)) < 0.f;
    }

    CPT_CPU_GPU static Vec3 snell_refraction(const Vec3& incid, const Vec3& normal, float& cos_r2, float dot_n, float ni, float nr) {
        /* Refraction vector by Snell's Law, note that an extra flag will be returned */
        float ratio = ni / nr;
        cos_r2 = 1.f - (ratio * ratio) * (1. - dot_n * dot_n);        // refraction angle cosine
        // for ni > nr situation, there will be total reflection
        // if cos_r2 <= 0.f, then return value will be Vec3(0, 0, 0)
        return (ratio * incid - ratio * dot_n * normal + sgn(dot_n) * sqrtf(fabsf(cos_r2)) * normal).normalized() * (cos_r2 > 0.f);
    }

    CPT_CPU_GPU static float fresnel_equation(float n_in, float n_out, float cos_inc, float cos_ref) {
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