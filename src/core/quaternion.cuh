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
 * @file quaternion.cuh
 * @author Qianyue He
 * @brief Implement simple Quaternion to avoid Gimbal lock
 * @date 9.17.2024
 * @copyright Copyright (c) 2024
 */
#pragma once
#include "core/vec3.cuh"

class Quaternion {
  public:
    float x, y, z, w;

    CPT_CPU_GPU Quaternion() : x(0), y(0), z(0), w(1) {}
    CPT_CPU_GPU Quaternion(float x, float y, float z, float w)
        : x(x), y(y), z(z), w(w) {}

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU static Quaternion angleAxis_host(float angle, VecType &&v) {
        // v is guaranteed to be normalized
        // half sin and cos
        float half_angle = angle * 0.5f;
        float sin_half_angle = sinf(half_angle);
        float cos_half_angle = cosf(half_angle);
        return Quaternion(v.x() * sin_half_angle, v.y() * sin_half_angle,
                          v.z() * sin_half_angle, cos_half_angle);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_GPU_INLINE static Quaternion angleAxis(float angle, VecType &&v) {
        float half_angle = angle * 0.5f;
        float sin_half_angle, cos_half_angle;
        sincosf(half_angle, &sin_half_angle, &cos_half_angle);
        return Quaternion(v.x() * sin_half_angle, v.y() * sin_half_angle,
                          v.z() * sin_half_angle, cos_half_angle);
    }

    // quat multiplication
    CPT_CPU_GPU_INLINE Quaternion operator*(const Quaternion &q) const {
        return Quaternion(w * q.x + x * q.w + y * q.z - z * q.y, // x components
                          w * q.y - x * q.z + y * q.w + z * q.x, // y components
                          w * q.z + x * q.y - y * q.x + z * q.w, // z components
                          w * q.w - x * q.x - y * q.y - z * q.z  // w components
        );
    }

    // normalize
    CPT_GPU_INLINE void normalize() {
        float len = rsqrtf(x * x + y * y + z * z + w * w);
        x *= len;
        y *= len;
        z *= len;
        w *= len;
    }

    CPT_CPU_GPU_INLINE Quaternion conjugate() const {
        return Quaternion(-x, -y, -z, w);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE static Quaternion from_vector(VecType &&v) {
        return Quaternion(v.x(), v.y(), v.z(), 0.0f);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE Vec3 rotate(VecType &&v) const {
        Quaternion v_quat = Quaternion::from_vector(std::forward<VecType>(v));
        Quaternion qv = (*this) * v_quat;
        Quaternion rotated_quat = qv * this->conjugate();
        return Vec3(rotated_quat.x, rotated_quat.y, rotated_quat.z);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE Vec3 inverse_rotate(VecType &&v) const {
        Quaternion v_quat = Quaternion::from_vector(std::forward<VecType>(v));
        Quaternion qv = this->conjugate() * v_quat;
        Quaternion rotated_quat = qv * (*this);
        return Vec3(rotated_quat.x, rotated_quat.y, rotated_quat.z);
    }
};
