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
#include "core/textures.cuh"
#include "core/interaction.cuh"
#include "core/preset_params.cuh"

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
    Dispersion     = 0x06,
    NumSupportedBSDF = 0x07
};

extern const std::array<const char*, NumSupportedBSDF> BSDF_NAMES;

class BSDF {
public:
    Vec4 k_d;
    Vec4 k_s;
    Vec4 k_g;
    int bsdf_flag;
    int __padding;
public:
    CPT_CPU_GPU BSDF() {}
    CPT_CPU_GPU BSDF(Vec4 _k_d, Vec4 _k_s, Vec4 _k_g, int flag = BSDFFlag::BSDF_NONE):
        k_d(std::move(k_d)), k_s(std::move(_k_s)), k_g(std::move(_k_g)), bsdf_flag(flag)
    {}

    CPT_GPU void set_kd(Vec4&& v) noexcept { this->k_d = v; }
    CPT_GPU void set_ks(Vec4&& v) noexcept { this->k_s = v; }
    CPT_GPU void set_kg(Vec4&& v) noexcept { this->k_g = v; }
    CPT_GPU void set_lobe(int v) noexcept { this->bsdf_flag = v; }

    CPT_GPU virtual float pdf(const Interaction& it, const Vec3& out, const Vec3& in, int index) const = 0;

    CPT_GPU virtual Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const = 0;

    CPT_GPU virtual Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
    ) const = 0;

    CPT_GPU_INLINE bool require_lobe(BSDFFlag flags) const noexcept {
        return (bsdf_flag & (int)flags) > 0;
    }
};

class LambertianBSDF: public BSDF {
public:
    using BSDF::k_d;
    using BSDF::bsdf_flag;
    CPT_CPU_GPU LambertianBSDF(Vec4 _k_d, int kd_id = -1):
        BSDF(std::move(_k_d), Vec4(0, 0, 0), Vec4(0, 0, 0), BSDFFlag::BSDF_DIFFUSE | BSDFFlag::BSDF_REFLECT) {}

    CPT_CPU_GPU LambertianBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */, int index) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        float dot_val = normal.dot(out);
        return max(normal.dot(out), 0.f) * M_1_Pi;
    }

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        float cos_term = normal.dot(out);
        float dot_in  = normal.dot(in);
        float same_side = (dot_in > 0) ^ (cos_term > 0);     // should be positive or negative at the same time
        const cudaTextureObject_t diff_tex = c_textures.diff_tex[index];
        return c_textures.eval(diff_tex, it.uv_coord, k_d) * max(0.f, cos_term) * M_1_Pi * same_side;
    }

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
    ) const override {
        auto local_ray = sample_cosine_hemisphere(sp.next2D(), pdf);
        const Vec3 normal = c_textures.eval_normal(it, index);
        auto out_ray = delocalize_rotate(normal, local_ray);
        // throughput *= f / pdf --> k_d * cos / pi / (pdf = cos / pi) == k_d
        float dot_in  = normal.dot(indir);
        float dot_out = normal.dot(out_ray);
        const cudaTextureObject_t diff_tex = c_textures.diff_tex[index];
        throughput *= c_textures.eval(diff_tex, it.uv_coord, k_d) * ((dot_in > 0) ^ (dot_out > 0));
        samp_lobe = static_cast<BSDFFlag>(bsdf_flag);
        return out_ray;
    }
};

class SpecularBSDF: public BSDF {
public:
    using BSDF::k_s;
    CPT_CPU_GPU SpecularBSDF(Vec4 _k_s):
        BSDF(Vec4(0, 0, 0), std::move(_k_s), Vec4(0, 0, 0), BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_REFLECT) {}

    CPT_CPU_GPU SpecularBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */, int) const override {
        return 0.f;
    }

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        auto ref_dir = in.advance(normal, -2.f * in.dot(normal)).normalized();
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        return c_textures.eval(spec_tex, it.uv_coord, k_s) * (out.dot(ref_dir) > 0.99999f);
    }

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
    ) const override {
        // throughput *= f / pdf
        samp_lobe = static_cast<BSDFFlag>(bsdf_flag);
        const Vec3 normal = c_textures.eval_normal(it, index);
        float in_dot_n = indir.dot(normal);
        pdf = 1.f;
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        throughput *= c_textures.eval(spec_tex, it.uv_coord, k_s);
        return -reflection(indir, normal, in_dot_n);
    }
};

class TranslucentBSDF: public BSDF {
public:
    using BSDF::k_s;        // specular reflection
    using BSDF::k_d;        // ior

    CPT_CPU_GPU TranslucentBSDF(Vec4 k_s, Vec4 ior):
        BSDF(std::move(ior), std::move(k_s), Vec4(0, 0, 0), BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_TRANSMIT) {}

    CPT_CPU_GPU TranslucentBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& incid, int) const override {
        return 0.f;
    }

    CPT_GPU_INLINE static Vec4 eval_impl(
        const Vec3& normal, 
        const Vec3& out, 
        const Vec3& in,
        const Vec4& ks,
        const float eta,
        bool is_radiance = true
    ) {
        float dot_normal = in.dot(normal);
        // at least according to pbrt-v3, ni / nr is computed as the following (using shading normal)
        // see https://computergraphics.stackexchange.com/questions/13540/shading-normal-and-geometric-normal-for-refractive-surface-rendering
        float ni = dot_normal < 0 ? 1.f : eta;
        float nr = dot_normal < 0 ? eta : 1.f, cos_r2 = 0;
        float eta2 = ni * ni / (nr * nr);
        Vec3 ret_dir = in.advance(normal, -2.f * dot_normal).normalized(),
             refra_vec = FresnelTerms::snell_refraction(in, normal, cos_r2, dot_normal, ni, nr);
        bool total_ref = FresnelTerms::is_total_reflection(dot_normal, ni, nr);
        nr = FresnelTerms::fresnel_dielectric(ni, nr, fabsf(dot_normal), sqrtf(fabsf(cos_r2)));
        bool reflc_dot = out.dot(ret_dir) > 0.99999f, refra_dot = out.dot(refra_vec) > 0.99999f;        // 0.9999  means 0.26 deg
        return ks * (reflc_dot | refra_dot) * (refra_dot && is_radiance ? eta2 : 1.f);
    } 

    CPT_GPU_INLINE static Vec3 sample_dir_impl(
        const Vec3& indir, 
        const Vec3& normal, 
        const Vec4& ks,
        const float eta,
        Vec4& throughput,
        Sampler& sp,
        float& pdf,
        BSDFFlag& samp_lobe, 
        bool is_radiance = true
    ) {
        float dot_normal = indir.dot(normal);
        // at least according to pbrt-v3, ni / nr is computed as the following (using shading normal)
        // see https://computergraphics.stackexchange.com/questions/13540/shading-normal-and-geometric-normal-for-refractive-surface-rendering
        float ni = dot_normal < 0 ? 1.f : eta;
        float nr = dot_normal < 0 ? eta : 1.f, cos_r2 = 0;
        float eta2 = ni * ni / (nr * nr);
        Vec3 ret_dir = indir.advance(normal, -2.f * dot_normal).normalized(),
             refra_vec = FresnelTerms::snell_refraction(indir, normal, cos_r2, dot_normal, ni, nr);
        bool total_ref = FresnelTerms::is_total_reflection(dot_normal, ni, nr);
        nr = FresnelTerms::fresnel_dielectric(ni, nr, fabsf(dot_normal), sqrtf(fabsf(cos_r2)));
        bool reflect = total_ref || sp.next1D() < nr;
        ret_dir = select(ret_dir, refra_vec, reflect);
        pdf     = total_ref ? 1.f : (reflect ? nr : 1.f - nr);
        samp_lobe = static_cast<BSDFFlag>(
            BSDFFlag::BSDF_SPECULAR | (total_ref || reflect ? BSDFFlag::BSDF_REFLECT : BSDFFlag::BSDF_TRANSMIT)
        );
        throughput *= ks * (is_radiance && !reflect ? eta2 : 1.f);
        return ret_dir;
    }

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        const Vec4 ks = c_textures.eval(spec_tex, it.uv_coord, k_s);
        float eta = c_textures.eval_rough(it.uv_coord, index, Vec2( k_d.x() )).x();
        return eval_impl(normal, out, in, ks, eta, is_radiance);
    }

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
    ) const override {
        const Vec3 normal = c_textures.eval_normal(it, index);
        const cudaTextureObject_t spec_tex = c_textures.spec_tex[index];
        const Vec4 ks = c_textures.eval(spec_tex, it.uv_coord, k_s);
        float eta = c_textures.eval_rough(it.uv_coord, index, Vec2( k_d.x() )).x();
        return sample_dir_impl(indir, normal, ks, eta, throughput, sp, pdf, samp_lobe, is_radiance);
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
    CPT_CPU_GPU PlasticBSDF(Vec4 _k_d, Vec4 _k_s, Vec4 sigma_a, 
        float ior, float trans_scaler = 1.f, float thickness = 0
    );

    CPT_CPU_GPU PlasticBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */, int) const override;

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
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
    CPT_CPU_GPU PlasticForwardBSDF(Vec4 _k_d, Vec4 _k_s, Vec4 sigma_a, 
        float ior, float trans_scaler = 1.f, float thickness = 0
    );

    CPT_CPU_GPU PlasticForwardBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */, int index) const override;

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
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
    CPT_CPU_GPU GGXConductorBSDF(Vec3 eta_t, Vec3 k, Vec4 albedo, float roughness_x, float roughness_y);

    CPT_CPU_GPU GGXConductorBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& /* in */, int index) const override;

    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override;

    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
    ) const override;
};

/**
 * @brief 360nm to 830nm Wave dispersion translucent BSDF
 * We uniformly sample from wavelength range 360nm to 830nm,
 * and the IoR is computed by Cauchy's Equation A + B / \lambda^2
 */
class DispersionBSDF: public BSDF {
public:
    static constexpr float WL_MIN   = 360;
    static constexpr float WL_RANGE = 471;        // 360 + 471 -> 831 (830 included for indexing)
    static constexpr float D65_MIN   = 300;
    static constexpr float D65_RANGE = 531;
public:
    CPT_CPU_GPU DispersionBSDF(Vec4 k_s, float index_a, float index_b):
        BSDF(Vec4(index_a, index_b, 0), std::move(k_s), Vec4(0, 0, 0), BSDFFlag::BSDF_DIFFUSE | BSDFFlag::BSDF_TRANSMIT) {}

    CPT_CPU_GPU DispersionBSDF(): BSDF() {}
    
    CPT_GPU float pdf(const Interaction& it, const Vec3& out, const Vec3& incid, int) const override;
    CPT_GPU Vec4 eval(const Interaction& it, const Vec3& out, const Vec3& in, int index, bool is_mi = false, bool is_radiance = true) const override;
    CPT_GPU Vec3 sample_dir(
        const Vec3& indir, const Interaction& it, Vec4& throughput, float& pdf, 
        Sampler& sp, BSDFFlag& samp_lobe, int index, bool is_radiance = true
    ) const override;

    CPT_GPU_INLINE static Vec4 wavelength_to_XYZ(float wavelength);
    CPT_GPU_INLINE static Vec4 wavelength_to_RGB(float wavelength);

    CPT_GPU_INLINE static float sample_wavelength(Sampler& sp) {
        return sp.next1D() * WL_RANGE + WL_MIN;
    }

    CPT_GPU_INLINE float get_ior(float wavelength) const {
        // k_d.y() is B, (nm^2)
        return k_d.x() + k_d.y() / (wavelength * wavelength);
    }

    CONDITION_TEMPLATE_SEP_3(VType1, VType2, NType, Vec3, Vec3, Vec3)
    CPT_GPU_INLINE bool get_wavelength_from(VType1&& indir, VType2&& outdir, NType&& normal, float& wavelength) const {
        float cos_i = normal.dot(indir), cos_o = normal.dot(outdir),
              sin_i = sqrtf(1.f - cos_i * cos_i), sin_o = sqrtf(1.f - cos_o * cos_o);
        float eta = sin_i > sin_o ? sin_i / sin_o : sin_o / sin_i;
        wavelength = sqrtf(k_d.y() / fmaxf(eta - k_d.x(), 1e-5f));
        return wavelength > WL_MIN && wavelength < WL_MIN + WL_RANGE;
    }
};