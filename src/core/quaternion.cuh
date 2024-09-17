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
    CPT_CPU_GPU Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU static Quaternion angleAxis(float angle, VecType&& v) {
        // v is guaranteed to be normalized
        // half sin and cos
        float half_angle = angle * 0.5f;
        float sin_half_angle = sinf(half_angle);
        float cos_half_angle = cosf(half_angle);
        return Quaternion(
            v.x() * sin_half_angle,
            v.y() * sin_half_angle,
            v.z() * sin_half_angle,
            cos_half_angle
        );
    }

    // quat multiplication
    CPT_CPU_GPU Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            w * q.x + x * q.w + y * q.z - z * q.y,  // x components
            w * q.y - x * q.z + y * q.w + z * q.x,  // y components
            w * q.z + x * q.y - y * q.x + z * q.w,  // z components
            w * q.w - x * q.x - y * q.y - z * q.z   // w components
        );
    }

    // normalize
    CPT_CPU_GPU void normalize() {
        float len = sqrtf(x * x + y * y + z * z + w * w);
        x /= len;
        y /= len;
        z /= len;
        w /= len;
    }
};