// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * @author: Qianyue He
 * @brief Frequently used sampler
 * @date:   5.12.2024
 */
#pragma once
#include "core/sampler.cuh"
#include "core/so3.cuh"

CONDITION_TEMPLATE(VecType, Vec2)
CPT_CPU_GPU_INLINE Vec3 sample_cosine_hemisphere(VecType &&uv, float &pdf) {
    float cos_theta = sqrtf(uv.x());
    float sin_theta = sqrtf(1. - uv.x());
    pdf = cos_theta * M_1_Pi;
    float sin_phi = 0, cos_phi = 0;
    sincospif(2.f * uv.y(), &sin_phi, &cos_phi);
    return Vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

CONDITION_TEMPLATE(VecType, Vec2)
CPT_CPU_GPU_INLINE Vec3 sample_uniform_sphere(VecType &&uv, float &pdf) {
    float cos_theta = 2.f * uv.x() - 1.f;
    float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    pdf = 0.25f * M_1_Pi;
    float sin_phi = 0, cos_phi = 0;
    sincospif(2.f * uv.y(), &sin_phi, &cos_phi);
    return Vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

CONDITION_TEMPLATE(VecType, Vec2)
CPT_CPU_GPU_INLINE Vec3 sample_uniform_cone(VecType &&uv, float cos_val,
                                            float &pdf) {
    float cos_theta =
        cos_val + (1.f - cos_val) * uv.x(); // uniform in [cos_val, 1]
    float sin_theta = sqrtf(1.f - cos_theta * cos_theta);
    pdf = 1.f / (2.f * M_Pi * (1.f - cos_val));
    float sin_phi = 0, cos_phi = 0;
    sincospif(2.f * uv.y(), &sin_phi, &cos_phi);

    return Vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

CONDITION_TEMPLATE(VecType, Vec2)
CPT_CPU_GPU_INLINE Vec2 sample_uniform_disk(VecType &&uv) {
    // non concentric simple 2D disk sampling
    float r = sqrtf(uv.x());
    float sin_theta = 0, cos_theta = 0;
    sincospif(2.f * uv.y(), &sin_theta, &cos_theta);
    return Vec2(r * cos_theta, r * sin_theta);
}
