/**
 * Frequently used sampler
 * @author: Qianyue He
 * @date:   5.12.2024
*/
#pragma once
#include "core/so3.cuh"
#include "core/sampler.cuh"

CONDITION_TEMPLATE(VecType, Vec2)
CPT_CPU_GPU_INLINE Vec3 sample_cosine_hemisphere(VecType&& uv, float& pdf) {
    float cos_theta = sqrtf(uv.x());
    float sin_theta = sqrtf(1. - uv.x());
    pdf = cos_theta * M_1_Pi;
    float sin_phi = 0, cos_phi = 0;
    sincospif(2.f * uv.y(), &sin_phi, &cos_phi);
    return Vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

CONDITION_TEMPLATE(VecType, Vec2)
CPT_CPU_GPU_INLINE Vec3 sample_uniform_sphere(VecType&& uv, float& pdf) {
    float cos_theta = 2.f * uv.x() - 1.f;
    float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    pdf = 0.25f * M_1_Pi;
    float sin_phi = 0, cos_phi = 0;
    sincospif(2.f * uv.y(), &sin_phi, &cos_phi);
    return Vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

CONDITION_TEMPLATE(VecType, Vec2)
CPT_CPU_GPU_INLINE Vec3 sample_uniform_cone(VecType&& uv, float cos_val, float& pdf) {
    float cos_theta = cos_val + (1.f - cos_val) * uv.x();  // uniform in [cos_val, 1]
    float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    pdf = 1.f / (2.f * M_Pi * (1.f - cos_val));
    float sin_phi = 0, cos_phi = 0;
    sincospif(2.f * uv.y(), &sin_phi, &cos_phi);

    return Vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}