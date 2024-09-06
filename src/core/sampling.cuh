/**
 * Frequently used sampler
 * @author: Qianyue He
 * @date:   5.12.2024
*/
#pragma once
#include "core/so3.cuh"
#include "core/sampler.cuh"

CPT_CPU_GPU_INLINE Vec3 sample_cosine_hemisphere(Vec2 uv, float& pdf) {
    float cos_theta = sqrtf(uv.x());
    float sin_theta = sqrtf(1. - uv.x());
    float phi = M_Pi * 2.f * uv.y();
    pdf = cos_theta * M_1_Pi;
    
    return Vec3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
}

CPT_CPU_GPU_INLINE Vec3 sample_uniform_sphere(Vec2 uv, float& pdf) {
    float cos_theta = 2.f * uv.x() - 1.f;
    float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    float phi = uv.y() * static_cast<float>(2.f * M_Pi);
    pdf = 0.25f * M_1_Pi;
    return Vec3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
}