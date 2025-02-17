#pragma once
/**
 * @file homogeneous.cuh
 * @author Qianyue He
 * @brief Homoegeneous scattering medium
 * @version 0.1
 * @date 2025-02-05
 * @copyright Copyright (c) 2025
 */
#include "core/medium.cuh"

// FIXME：no need to inline the functions since virtual functions won't be inlined (base class ptr)
class HomogeneousMedium: public Medium {
private:
    Vec4 sigma_a;
    Vec4 sigma_s;
    Vec4 sigma_t;
public:
    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec4, Vec4)
    CPT_CPU_GPU HomogeneousMedium(VType1&& _sigma_a, VType2&& _sigma_s):
        sigma_a(std::forward<VType1>(_sigma_a)), sigma_s(std::forward<VType2>(_sigma_s))
    {
        sigma_t = sigma_a + sigma_s;
    }

    CPT_GPU_INLINE MediumSample sample(const Ray& ray, Sampler& sp, float max_dist = MAX_DIST) const override {
        MediumSample msp;
        int channel = sp.discrete1D() % 3;
        msp.dist = -logf(1.f - sp.next1D()) / sigma_t[channel];
        bool is_medium = msp.dist < ray.hit_t;
        msp.dist = fminf(msp.dist, ray.hit_t);

        float exp_y_x = expf(-(sigma_t.y() - sigma_t.x()) * msp.dist),
              exp_z_x = expf(-(sigma_t.z() - sigma_t.x()) * msp.dist),
              exp_z_y = expf(-(sigma_t.z() - sigma_t.y()) * msp.dist),
              exp_x_y = __frcp_rn(exp_y_x),
              exp_x_z = __frcp_rn(exp_z_x),
              exp_y_z = __frcp_rn(exp_z_y);
        /**
         * Why so complicated?
         * 
         * If you implement in the similar way as PBRT-v3
         * You will find that when PDF is pretty small, you will need to clip it to prevent
         * numerical instability, but this can be detrimental to rendering results: unexpected
         * stripes will apear in the media. If you choose not to clip the results, you might wind
         * up having black dots. 
         * 
         * My following implementation directly get rid of the
         * need to clip, while also being numerically stable 
         */
        msp.local_thp = is_medium ? 
            Vec4(
                3.f / (sigma_t.x() + sigma_t.y() * exp_y_x + sigma_t.z() * exp_z_x),
                3.f / (sigma_t.x() * exp_x_y + sigma_t.y() + sigma_t.z() * exp_z_y),
                3.f / (sigma_t.x() * exp_x_z + sigma_t.y() * exp_y_z + sigma_t.z())
            ) * sigma_s : 
            Vec4(
                3.f / (1.f + exp_y_x + exp_z_x),
                3.f / (exp_x_y + 1.f + exp_z_y),
                3.f / (exp_x_z + exp_y_z + 1.f)
            );
        
        msp.flag = uint32_t(is_medium);
        return msp;
    }

    CPT_GPU_INLINE Vec4 transmittance(const Ray& ray, Sampler& /* sp */, float dist) const override {
        return (-sigma_t * dist).exp_xyz();
    }

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec4, Vec4)
    CPT_GPU_INLINE void set_params(VType1&& _sig_a, VType2&& _sig_s) {
        sigma_t = _sig_a + _sig_s;
        sigma_a = std::forward<VType1>(_sig_a);
        sigma_s = std::forward<VType2>(_sig_s);
    }
};

CPT_KERNEL void create_homogeneous_volume(
    Medium** media,
    PhaseFunction** phases,
    int med_id, int ph_id,
    Vec4 sigma_a, 
    Vec4 sigma_s,
    float scale
);