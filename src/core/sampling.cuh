/**
 * Frequently used sampler
 * @author: Qianyue He
 * @date:   5.12.2024
*/
#pragma once
#include "core/so3.cuh"
#include "core/sampler.cuh"

CPT_CPU_GPU Vec3 sample_cosine_hemisphere(Vec2&& uv, float& pdf) {
    float cos_theta = sqrtf(uv.x());
    float sin_theta = sqrtf(1. - uv.x());
    float phi = M_PI * 2.f * uv.y();
    pdf = cos_theta * M_1_PI;
    
    return Vec3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
}