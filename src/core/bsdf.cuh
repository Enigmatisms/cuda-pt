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
protected:
    Vec3 k_d;
    Vec3 k_s;
    Vec3 k_g;
    int kd_tex_id;
    int ex_tex_id;
public:
    CPT_CPU_GPU BSDF(Vec3 _k_d, Vec3 _k_s, Vec3 _k_g, int kd_id = -1, int kg_id = -1):
        k_d(std::move(k_d)), k_s(std::move(_k_s)), k_g(std::move(_k_g)),
        kd_tex_id(kd_id), ex_tex_id(kg_id)
    {}

    CPT_CPU_GPU virtual float pdf(const Interaction& it, const Vec3& out, const Vec3& in) const = 0;

    CPT_CPU_GPU virtual Vec3 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false) const = 0;

    CPT_CPU_GPU virtual Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec3& throughput, Vec2&& uv, float& sample_pdf) const = 0;
};


class LambertianBSDF: public BSDF {
using BSDF::k_d;
public:
    CPT_CPU_GPU LambertianBSDF(Vec3 _k_d, int kd_id = -1):
        BSDF(_k_d, Vec3(0, 0, 0), Vec3(0, 0, 0), kd_id, -1) {}

    CPT_CPU_GPU LambertianBSDF(Vec3 _k_d, Vec3 _k_s, Vec3 _k_g, int kd_id = -1, int kg_id = -1):
        BSDF(std::move(_k_d), std::move(_k_s), std::move(_k_g), kd_id, kg_id) {}
    
    CPT_CPU_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override {
        // printf("it.norm: %f, %f, %f\n", it.shading_norm.x(), it.shading_norm.y(), it.shading_norm.z());
        // printf("out: %f, %f, %f\n", out.x(), out.y(), out.z());
        auto res = it.shading_norm.dot(out) * M_1_PI;
    }

    CPT_CPU_GPU Vec3 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false) const override {
        float cosine_term = max(0.f, it.shading_norm.dot(out));
        return k_d * cosine_term * M_1_PI;
    }

    CPT_CPU_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec3& throughput, Vec2&& uv, float& sample_pdf) const override {
        auto local_ray = sample_cosine_hemisphere(std::move(uv), sample_pdf);
        throughput = max(0.f, local_ray.z()) * k_d * M_1_PI;
        return delocalize_rotate(Vec3(0, 0, 1), it.shading_norm, local_ray);;
    }
};

class SpecularBSDF: public BSDF {
using BSDF::k_s;
public:
    CPT_CPU_GPU SpecularBSDF(Vec3 _k_s, int ks_id = -1):
        BSDF(Vec3(0, 0, 0), _k_s, Vec3(0, 0, 0), -1, ks_id) {}

    CPT_CPU_GPU SpecularBSDF(Vec3 _k_d, Vec3 _k_s, Vec3 _k_g, int kd_id = -1, int kg_id = -1):
        BSDF(std::move(_k_d), std::move(_k_s), std::move(_k_g), kd_id, kg_id) {}
    
    CPT_CPU_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override {
        return 0.f;
    }

    CPT_CPU_GPU Vec3 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false) const override {
        return Vec3(0, 0, 0);
    }

    CPT_CPU_GPU Vec3 sample_dir(const Vec3& indir, const Interaction& it, Vec3& throughput, Vec2&& uv, float& sample_pdf) const override {
        sample_pdf = 1;
        float cosine_term = indir.dot(it.shading_norm);
        throughput = k_s * max(-cosine_term, 0.f);
        return indir - 2.f * indir.dot(it.shading_norm) * it.shading_norm;
    }
};