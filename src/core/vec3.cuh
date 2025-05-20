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
 * @brief CUDA/CPU 3D vector implementation
 * @author: Qianyue He
 * @date:   4.10.2024
 */
#pragma once
#include "core/constants.cuh"
#include "core/cuda_utils.cuh"
#include <chrono>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>

class Vec3 {
  private:
    float3 _data;

  public:
    CPT_CPU_GPU Vec3() {}
    CPT_CPU_GPU
    constexpr Vec3(float _x, float _y, float _z) : _data({_x, _y, _z}) {}

    CPT_CPU_GPU
    constexpr explicit Vec3(float _v) : _data({_v, _v, _v}) {}

    CPT_CPU_GPU
    Vec3(float3 &&v) : _data(std::move(v)) {}

    CPT_CPU_GPU_INLINE
    float &operator[](int index) { return *((&_data.x) + index); }

    CPT_CPU_GPU_INLINE
    float operator[](int index) const { return *((&_data.x) + index); }

    CPT_CPU_GPU_INLINE float &x() { return _data.x; }
    CPT_CPU_GPU_INLINE float &y() { return _data.y; }
    CPT_CPU_GPU_INLINE float &z() { return _data.z; }

    constexpr CPT_CPU_GPU_INLINE const float &x() const { return _data.x; }
    constexpr CPT_CPU_GPU_INLINE const float &y() const { return _data.y; }
    constexpr CPT_CPU_GPU_INLINE const float &z() const { return _data.z; }

    CPT_CPU_GPU_INLINE
    Vec3 abs() const noexcept {
        return Vec3(fabs(_data.x), fabs(_data.y), fabs(_data.z));
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    constexpr Vec3 operator+(VecType &&b) const noexcept {
        return Vec3(_data.x + b.x(), _data.y + b.y(), _data.z + b.z());
    }

    CPT_CPU_GPU_INLINE
    constexpr Vec3 operator+(float b) const noexcept {
        return Vec3(_data.x + b, _data.y + b, _data.z + b);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 &operator+=(VecType &&b) noexcept {
        _data.x += b.x();
        _data.y += b.y();
        _data.z += b.z();
        return *this;
    }

    CPT_CPU_GPU_INLINE
    Vec3 &operator+=(float v) noexcept {
        _data.x += v;
        _data.y += v;
        _data.z += v;
        return *this;
    }

    CPT_CPU_GPU_INLINE
    Vec3 operator-() const noexcept {
        return Vec3(-_data.x, -_data.y, -_data.z);
    }

    CPT_CPU_GPU_INLINE
    constexpr Vec3 operator-(float b) const noexcept {
        return Vec3(_data.x - b, _data.y - b, _data.z - b);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 &operator-=(VecType &&b) noexcept {
        _data.x -= b.x();
        _data.y -= b.y();
        _data.z -= b.z();
        return *this;
    }

    CPT_CPU_GPU_INLINE
    Vec3 &operator-=(float v) noexcept {
        _data.x -= v;
        _data.y -= v;
        _data.z -= v;
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    constexpr Vec3 operator-(VecType &&b) const {
        return Vec3(_data.x - b.x(), _data.y - b.y(), _data.z - b.z());
    }

    CPT_CPU_GPU_INLINE
    constexpr Vec3 operator*(float b) const noexcept {
        return Vec3(_data.x * b, _data.y * b, _data.z * b);
    }

    CPT_CPU_GPU_INLINE
    Vec3 &operator*=(float b) noexcept {
        _data.x *= b;
        _data.y *= b;
        _data.z *= b;
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    constexpr Vec3 operator/(VecType &&b) const noexcept {
        return Vec3(_data.x / b.x(), _data.y / b.y(), _data.z / b.z());
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    constexpr Vec3 operator*(VecType &&b) const noexcept {
        return Vec3(_data.x * b.x(), _data.y * b.y(), _data.z * b.z());
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    void operator*=(VecType &&b) noexcept {
        _data.x *= b.x();
        _data.y *= b.y();
        _data.z *= b.z();
    }

    CPT_CPU_GPU_INLINE void fill(float v) noexcept {
        _data = make_float3(v, v, v);
    }

    CPT_CPU_GPU_INLINE
    bool numeric_err() const noexcept {
        return isnan(_data.x) || isnan(_data.y) || isnan(_data.z) ||
               isinf(_data.x) || isinf(_data.y) || isinf(_data.z);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 advance(VecType &&d, float t) const noexcept {
        return Vec3(fmaf(d.x(), t, _data.x), fmaf(d.y(), t, _data.y),
                    fmaf(d.z(), t, _data.z));
    }

    // only for local directions
    CPT_CPU_GPU_INLINE Vec3 face_forward() const noexcept {
        return _data.z > 0 ? *this : Vec3(-_data.x, -_data.y, -_data.z);
    }

    // ============== Specialized version using CUDA math function
    // ===============
    CPT_GPU_INLINE
    Vec3 normalized() const {
        return *this * rnorm3df(_data.x, _data.y, _data.z);
    }

    CPT_GPU_INLINE
    void normalize() { this->operator*=(rnorm3df(_data.x, _data.y, _data.z)); }

    CPT_GPU_INLINE
    float length() const { return norm3df(_data.x, _data.y, _data.z); }
    // ============== Specialized version using CUDA math function
    // ===============

    CPT_CPU_GPU_INLINE
    float length2() const {
        return fmaf(_data.x, _data.x,
                    fmaf(_data.y, _data.y, _data.z * _data.z));
    }

    CPT_CPU_INLINE
    Vec3 normalized_h() const { return *this * rsqrtf(length2()); }

    CPT_CPU_INLINE
    void normalize_h() { this->operator*=(rsqrtf(length2())); }

    CPT_CPU_INLINE
    float length_h() const { return sqrtf(length2()); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    float dot(VecType &&b) const noexcept {
        return fmaf(_data.x, b.x(), fmaf(_data.y, b.y(), _data.z * b.z()));
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 cross(VecType &&b) const noexcept {
        return Vec3(fmaf(_data.y, b.z(), -_data.z * b.y()),
                    fmaf(_data.z, b.x(), -_data.x * b.z()),
                    fmaf(_data.x, b.y(), -_data.y * b.x()));
    }

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 fmsub(VType1 &&b, VType2 &&c) const noexcept {
        return Vec3(fmaf(_data.x, b.x(), -c.x()), fmaf(_data.y, b.y(), -c.y()),
                    fmaf(_data.z, b.z(), -c.z()));
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 maximize(VecType &&b) const noexcept {
        return Vec3(fmaxf(_data.x, b.x()), fmaxf(_data.y, b.y()),
                    fmaxf(_data.z, b.z()));
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    void minimized(VecType &&b) noexcept {
        _data.x = fminf(_data.x, b.x());
        _data.y = fminf(_data.y, b.y());
        _data.z = fminf(_data.z, b.z());
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    void maximized(VecType &&b) noexcept {
        _data.x = fmaxf(_data.x, b.x());
        _data.y = fmaxf(_data.y, b.y());
        _data.z = fmaxf(_data.z, b.z());
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 minimize(VecType &&b) const noexcept {
        return Vec3(fminf(_data.x, b.x()), fminf(_data.y, b.y()),
                    fminf(_data.z, b.z()));
    }

    CPT_CPU_GPU_INLINE
    float max_elem() const noexcept {
        return fmaxf(_data.x, fmaxf(_data.y, _data.z));
    }

    CPT_CPU_GPU_INLINE
    float min_elem() const noexcept {
        return fminf(_data.x, fminf(_data.y, _data.z));
    }

    // Get the minimum value of the maximized vector, and the maximum value of
    // the minized vector
    CPT_GPU_INLINE
    void min_max(const Vec3 &input, float &mini, float &maxi) const noexcept {
        float min_x = _data.x < input.x() ? _data.x : input.x();
        float max_x = _data.x < input.x() ? input.x() : _data.x;

        float min_y = _data.y < input.y() ? _data.y : input.y();
        float max_y = _data.y < input.y() ? input.y() : _data.y;

        float min_z = _data.z < input.z() ? _data.z : input.z();
        float max_z = _data.z < input.z() ? input.z() : _data.z;

        mini = fmaxf(min_x, fmaxf(min_y, min_z));
        maxi = fminf(max_x, fminf(max_y, max_z));
    }

    CPT_GPU_INLINE Vec3 rcp() const noexcept {
        return Vec3(__frcp_rn(_data.x), __frcp_rn(_data.y), __frcp_rn(_data.z));
    }

    CPT_CPU_GPU_INLINE operator float3() const { return _data; }
};

CPT_CPU_GPU_INLINE void print_vec3(const Vec3 &obj) {
    printf("[%f, %f, %f]\n", obj.x(), obj.y(), obj.z());
}

CONDITION_TEMPLATE(VecType, Vec3)
CPT_CPU_GPU_INLINE
Vec3 operator*(float b, VecType &&v) noexcept {
    return Vec3(v.x() * b, v.y() * b, v.z() * b);
}

CONDITION_TEMPLATE(VecType, Vec3)
CPT_GPU_INLINE
Vec3 operator/(float b, VecType &&v) noexcept {
    return Vec3(b * __frcp_rn(v.x()), b * __frcp_rn(v.y()),
                b * __frcp_rn(v.z()));
}

CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
CPT_GPU_INLINE
Vec3 reflection(VType1 &&ray, VType2 &&normal, float dot_prod) noexcept {
    return 2.f * dot_prod * normal - ray;
}

CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
CPT_GPU_INLINE
Vec3 reflection(VType1 &&ray, VType2 &&normal) noexcept {
    return reflection(std::forward<VType1>(ray), std::forward<VType1>(normal),
                      ray.dot(normal));
}
