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

class HomogeneousMedium: public Medium {
private:
    Vec4 sigma_a;
    Vec4 sigma_s;
    Vec4 sigma_t;

    CONDITION_TEMPLATE_2(VType1, VType2, Vec4, Vec4)
    CPT_CPU_GPU HomogeneousMedium(VType1&& _sigma_a, VType2&& _sigma_s):
        sigma_a(std::forward<VType1>(_sigma_a)), sigma_s(std::forward<VType2>(sigma_s))
    {
        sigma_t = sigma_a + sigma_s;
    }

    CPT_GPU_INLINE MediumSample sample(const Ray& ray, Sampler& sp, float max_dist = MAX_DIST) const override {
        MediumSample msp;
        int channel = sp.discrete1D() % 3;
        msp.dist = -logf(1.f - sp.next1D()) / sigma_t[channel];
        bool is_medium = msp.dist < ray.hit_t;
        msp.dist = fminf(msp.dist, ray.hit_t);
        msp.local_thp = (-sigma_t * msp.dist).exp_xyz();
        Vec4 density = is_medium ? msp.local_thp * sigma_t : msp.local_thp;
        msp.pdf = (1.f / 3.f) * (density.x() + density.y() + density.z());
        msp.pdf = msp.pdf > 1e-6f ? msp.pdf : 1.f;
        msp.local_thp *= 1.f / msp.pdf;
        msp.local_thp = is_medium ? msp.local_thp * sigma_s : msp.local_thp;
        msp.flag = uint32_t(is_medium);
        return msp;
    }

    CPT_GPU_INLINE Vec4 HomogeneousMedium::transmittance(const Ray& ray, float dist) const override {
        return (-sigma_t * dist).exp_xyz();
    }
};