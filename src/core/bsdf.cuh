/**
 * CUDA object information
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include <array>
#include "core/vec2.cuh"
#include "core/vec4.cuh"
#include "core/fresnel.cuh"
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

enum BSDFType: uint8_t {
    Lambertian     = 0x00,
    Specular       = 0x01,
    Translucent    = 0x02,
    Plastic        = 0x03,
    PlasticForward = 0x04,
    GGXConductor   = 0x05,
    NumSupportedBSDF = 0x06
};

extern const std::array<const char*, NumSupportedBSDF> BSDF_NAMES;

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

    CPT_GPU virtual Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, bool is_radiance = true
    ) const = 0;

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
public:
    using BSDF::k_d;
    using BSDF::bsdf_flag;
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

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, bool is_radiance = true
    ) const override {
        auto local_ray = sample_cosine_hemisphere(sp.next2D(), pdf);
        auto out_ray = delocalize_rotate(it.shading_norm, local_ray);
        // throughput *= f / pdf --> k_d * cos / pi / (pdf = cos / pi) == k_d
        float dot_in  = it.shading_norm.dot(indir);
        float dot_out = it.shading_norm.dot(out_ray);
        throughput *= k_d * ((dot_in > 0) ^ (dot_out > 0));
        samp_lobe = static_cast<BSDFFlag>(bsdf_flag);
        return out_ray;
    }
};

class SpecularBSDF: public BSDF {
public:
    using BSDF::k_s;
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

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, bool is_radiance = true
    ) const override {
        // throughput *= f / pdf
        samp_lobe = static_cast<BSDFFlag>(bsdf_flag);
        float in_dot_n = indir.dot(it.shading_norm);
        pdf = 1.f;
        throughput *= k_s;
        return -reflection(indir, it.shading_norm, in_dot_n);
    }
};

class TranslucentBSDF: public BSDF {
public:
    using BSDF::k_s;        // specular reflection
    using BSDF::k_d;        // ior

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

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, bool is_radiance = true
    ) const override {
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
        samp_lobe = static_cast<BSDFFlag>(
            BSDFFlag::BSDF_SPECULAR | (total_ref || reflect ? BSDFFlag::BSDF_REFLECT : BSDFFlag::BSDF_TRANSMIT)
        );

        throughput *= k_s * (is_radiance && !reflect ? eta2 : 1.f);
        return ret_dir;
    }
};

class PlasticBSDF: public BSDF {
public:
    float trans_scaler;
    float thickness;
    float eta;
    float precomp_diff_f;       // precomputed diffuse Fresnel

    using BSDF::k_s;
    using BSDF::k_d;
    using BSDF::k_g;
public:
    CPT_CPU_GPU PlasticBSDF(Vec4 _k_d, Vec4 _k_s, Vec4 sigma_a, float ior, 
        float trans_scaler = 1.f, float thickness = 0, int kd_id = -1, int ks_id = -1
    );

    CPT_CPU_GPU PlasticBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override;

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false, bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, bool is_radiance = true
    ) const override;
};

/**
 * @brief specular reflection and delta forward
 */
class PlasticForwardBSDF: public BSDF {
public:
    float trans_scaler;
    float thickness;
    float eta;

    using BSDF::k_s;
    using BSDF::k_d;
    using BSDF::k_g;
public:
    CPT_CPU_GPU PlasticForwardBSDF(Vec4 _k_d, Vec4 _k_s, Vec4 sigma_a, float ior, 
        float trans_scaler = 1.f, float thickness = 0, int kd_id = -1, int ks_id = -1
    );

    CPT_CPU_GPU PlasticForwardBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override;

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false, bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, bool is_radiance = true
    ) const override;
};

class GGXConductorBSDF: public BSDF {
/**
 * @brief GGX microfacet normal distribution based BSDF
 * k_d is the eta_t of the metal
 * k_s is the k (Vec3) and the mapped roughness (k_s[3])
 * k_g is the underlying color (albedo)
 */
public:
    using BSDF::k_s;
    FresnelTerms fresnel;
public:
    CPT_CPU_GPU GGXConductorBSDF(Vec3 eta_t, Vec3 k, Vec4 albedo, float roughness_x, float roughness_y, int ks_id = -1);

    CPT_CPU_GPU GGXConductorBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */) const override;

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, bool is_mi = false, bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, bool is_radiance = true
    ) const override;
};