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
 * @brief CUDA/CPU 3D vector implementation
 * @date:   2024.4.10
 */
#pragma once
#include "core/constants.cuh"
#include "core/cuda_utils.cuh"
#include <chrono>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>

class Vec4 {
  private:
    float4 _data;

  public:
    CPT_CPU_GPU Vec4() {}

    CPT_CPU_GPU
    constexpr explicit Vec4(float _v) : _data({_v, _v, _v, _v}) {}

    CPT_CPU_GPU
    constexpr explicit Vec4(float _v, float alpha)
        : _data({_v, _v, _v, alpha}) {}

    CPT_CPU_GPU
    constexpr Vec4(float _x, float _y, float _z, float _w = 1)
        : _data({_x, _y, _z, _w}) {}

    CPT_CPU_GPU
    Vec4(float4 &&v) : _data(std::move(v)) {}

    CPT_CPU_GPU
    Vec4(float3 &&v, float w) : _data(make_float4(v.x, v.y, v.z, w)) {}

    CPT_CPU_GPU_INLINE
    float &operator[](int index) { return *((&_data.x) + index); }

    CPT_CPU_GPU_INLINE
    float operator[](int index) const { return *((&_data.x) + index); }

    CPT_CPU_GPU_INLINE float &x() { return _data.x; }
    CPT_CPU_GPU_INLINE float &y() { return _data.y; }
    CPT_CPU_GPU_INLINE float &z() { return _data.z; }
    CPT_CPU_GPU_INLINE float &w() { return _data.w; }

    constexpr CPT_CPU_GPU_INLINE const float &x() const { return _data.x; }
    constexpr CPT_CPU_GPU_INLINE const float &y() const { return _data.y; }
    constexpr CPT_CPU_GPU_INLINE const float &z() const { return _data.z; }
    constexpr CPT_CPU_GPU_INLINE const float &w() const { return _data.w; }
    CPT_CPU_GPU_INLINE float3 xyz() const {
        return make_float3(_data.x, _data.y, _data.z);
    }

    CPT_CPU_GPU_INLINE
    Vec4 abs() const noexcept {
        return Vec4(fabs(_data.x), fabs(_data.y), fabs(_data.z), fabs(_data.w));
    }

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE
    constexpr Vec4 operator+(VecType &&b) const noexcept {
        return Vec4(_data.x + b.x(), _data.y + b.y(), _data.z + b.z(),
                    _data.w + b.w());
    }

    CPT_CPU_GPU_INLINE
    constexpr Vec4 operator+(float b) const noexcept {
        return Vec4(_data.x + b, _data.y + b, _data.z + b, _data.w + b);
    }

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE
    Vec4 &operator+=(VecType &&b) noexcept {
        _data.x += b.x();
        _data.y += b.y();
        _data.z += b.z();
        _data.w += b.w();
        return *this;
    }

    CPT_CPU_GPU_INLINE
    Vec4 &operator+=(float v) noexcept {
        _data.x += v;
        _data.y += v;
        _data.z += v;
        _data.w += v;
        return *this;
    }

    CPT_CPU_GPU_INLINE
    constexpr Vec4 operator-() const noexcept {
        return Vec4(-_data.x, -_data.y, -_data.z, -_data.w);
    }

    CPT_CPU_GPU_INLINE
    constexpr Vec4 operator-(float b) const noexcept {
        return Vec4(_data.x - b, _data.y - b, _data.z - b, _data.w - b);
    }

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE
    Vec4 &operator-=(VecType &&b) noexcept {
        _data.x -= b.x();
        _data.y -= b.y();
        _data.z -= b.z();
        _data.w -= b.w();
        return *this;
    }

    CPT_CPU_GPU_INLINE
    Vec4 &operator-=(float v) noexcept {
        _data.x -= v;
        _data.y -= v;
        _data.z -= v;
        _data.w -= v;
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE
    constexpr Vec4 operator-(VecType &&b) const {
        return Vec4(_data.x - b.x(), _data.y - b.y(), _data.z - b.z(),
                    _data.w - b.w());
    }

    CPT_CPU_GPU_INLINE
    constexpr Vec4 operator*(float b) const noexcept {
        return Vec4(_data.x * b, _data.y * b, _data.z * b, _data.w * b);
    }

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE
    constexpr Vec4 operator/(VecType &&b) const noexcept {
        return Vec4(_data.x / b.x(), _data.y / b.y(), _data.z / b.z(),
                    _data.w / b.w());
    }

    CPT_CPU_GPU_INLINE
    Vec4 &operator*=(float b) noexcept {
        _data.x *= b;
        _data.y *= b;
        _data.z *= b;
        _data.w *= b;
        return *this;
    }

    CPT_CPU_GPU_INLINE float4 *float4_ptr() {
        return reinterpret_cast<float4 *>(this);
    }

    CPT_CPU_GPU_INLINE const float4 *float4_ptr() const {
        return reinterpret_cast<const float4 *>(this);
    }

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE
    constexpr Vec4 operator*(VecType &&b) const noexcept {
        return Vec4(_data.x * b.x(), _data.y * b.y(), _data.z * b.z(),
                    _data.w * b.w());
    }

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE
    void operator*=(VecType &&b) noexcept {
        _data.x *= b.x();
        _data.y *= b.y();
        _data.z *= b.z();
        _data.w *= b.w();
    }

    CPT_CPU_GPU_INLINE
    constexpr bool
    operator<(float v) const noexcept { // only compare the first 3
        return _data.x < v && _data.y < v && _data.z < v;
    }

    CPT_CPU_GPU_INLINE void fill(float v) noexcept {
        _data = make_float4(v, v, v, v);
    }

    CPT_CPU_GPU_INLINE
    bool numeric_err() const noexcept {
        return isnan(_data.x) || isnan(_data.y) || isnan(_data.z) ||
               isinf(_data.x) || isinf(_data.y) || isinf(_data.z);
    }

    CPT_CPU_GPU_INLINE
    Vec4 exp_xyz() const noexcept {
        return Vec4(expf(_data.x), expf(_data.y), expf(_data.z), 1);
    }

    CPT_CPU_GPU_INLINE
    Vec4 gamma_corr(float factor = 1.f / 2.2f) const noexcept {
        return Vec4(powf(fmaxf(0.f, _data.x), factor),
                    powf(fmaxf(0.f, _data.y), factor),
                    powf(fmaxf(0.f, _data.z), factor));
    }

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE
    Vec4 maximize(VecType &&b) const noexcept {
        return Vec4(max(_data.x, b.x()), max(_data.y, b.y()),
                    max(_data.z, b.z()), max(_data.w, b.w()));
    }

    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE
    Vec4 minimize(VecType &&b) const noexcept {
        return Vec4(min(_data.x, b.x()), min(_data.y, b.y()),
                    min(_data.z, b.z()), min(_data.w, b.w()));
    }

    CPT_CPU_GPU_INLINE
    float max_elem_3d() const noexcept {
        return max(_data.x, max(_data.y, _data.z));
    }

    CPT_CPU_GPU_INLINE
    float max_elem() const noexcept {
        return max(max(_data.w, _data.x), max(_data.y, _data.z));
    }

    CPT_CPU_GPU_INLINE
    float min_elem() const noexcept {
        return min(min(_data.w, _data.x), min(_data.y, _data.z));
    }

    CPT_CPU_GPU_INLINE
    bool good() const noexcept {
        return _data.x > THP_EPS || _data.y > THP_EPS || _data.z > THP_EPS;
    }

    CPT_CPU_GPU_INLINE operator float4() const { return _data; }
};

CONDITION_TEMPLATE(VecType, Vec4)
CPT_CPU_GPU_INLINE void print_Vec4(VecType &&obj) {
    printf("[%f, %f, %f, %f]\n", obj.x(), obj.y(), obj.z(), obj.w());
}

CONDITION_TEMPLATE(VecType, Vec4)
CPT_CPU_GPU_INLINE
Vec4 operator*(float b, VecType &&v) noexcept {
    return Vec4(v.x() * b, v.y() * b, v.z() * b, v.w() * b);
}

CONDITION_TEMPLATE(VecType, Vec4)
CPT_CPU_GPU_INLINE
Vec4 operator/(float b, VecType &&v) noexcept {
    return Vec4(b / v.x(), b / v.y(), b / v.z(), b / v.w());
}
