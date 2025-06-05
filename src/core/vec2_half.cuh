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
 * @brief CUDA/CPU 2D vector (Half) implementation
 * Note that Half Vector is only used in GPU, host has no access to half
 * @date:  2024.11.4
 */
#pragma once
#include "core/cuda_utils.cuh"
#include "vec2.cuh"
#include <cuda_fp16.h>

#define HALF2(v) (*(reinterpret_cast<half2 *>(&v)))
#define CONST_HALF2(v) (*(reinterpret_cast<const half2 *>(&v)))

class Vec2Half {
  private:
    half2 _data;

  public:
    CPT_CPU_GPU Vec2Half() {}

    CPT_CPU_GPU
    Vec2Half(float _v)
        : _data(make_half2(__float2half(_v), __float2half(_v))) {}

    CONDITION_TEMPLATE_DEFAULT(Half2Type, half2)
    CPT_CPU_GPU
    Vec2Half(Half2Type &&_h) : _data(std::forward<Half2Type>(_h)) {}

    CONDITION_TEMPLATE_DEFAULT(Vec2Type, Vec2)
    CPT_GPU
    Vec2Half(Vec2Type &&_h) : _data(_h.x(), _h.y()) {}

    CPT_CPU_GPU
    Vec2Half(float _x, float _y)
        : _data(make_half2(__float2half(_x), __float2half(_y))) {}

    CPT_GPU
    Vec2Half(half _v) : _data(make_half2(_v, _v)) {}

    CPT_GPU
    Vec2Half(half _x, half _y) : _data(make_half2(_x, _y)) {}

    CPT_GPU_INLINE
    half &operator[](int index) { return *((&_data.x) + index); }

    CPT_GPU_INLINE
    half operator[](int index) const { return *((&_data.x) + index); }

    CPT_CPU_GPU_INLINE half &x() { return _data.x; }
    CPT_CPU_GPU_INLINE half &y() { return _data.y; }

    CPT_CPU_GPU_INLINE float2 xy_float() const noexcept {
        return __half22float2(_data);
    }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE Vec2Half operator+(VecType &&b) const noexcept {
        return Vec2Half(_data.x + b.x(), _data.y + b.y());
    }

    CPT_GPU_INLINE Vec2Half operator+(float b) const noexcept {
        auto v = __float2half(b);
        return Vec2Half(_data.x + v, _data.y + v);
    }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    Vec2Half &operator+=(VecType &&b) noexcept {
        _data.x += b.x();
        _data.y += b.y();
        return *this;
    }

    CPT_GPU_INLINE
    Vec2Half operator-() const noexcept { return Vec2Half(-_data.x, -_data.y); }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE
    Vec2Half &operator-=(VecType &&b) noexcept {
        _data.x -= b.x();
        _data.y -= b.y();
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE
    Vec2Half operator-(VecType &&b) const {
        return Vec2Half(_data.x - b.x(), _data.y - b.y());
    }

    CPT_GPU_INLINE Vec2Half operator-(float b) const {
        auto v = __float2half(b);
        return Vec2Half(_data.x - v, _data.y - v);
    }

    CPT_GPU_INLINE
    Vec2Half operator*(float b) const noexcept {
        auto v = __float2half(b);
        return Vec2Half(_data.x * v, _data.y * v);
    }

    CPT_GPU_INLINE
    Vec2Half &operator*=(float b) noexcept {
        auto v = __float2half(b);
        _data.x *= v;
        _data.y *= v;
        return *this;
    }

    CPT_GPU_INLINE
    Vec2Half operator*(half b) const noexcept {
        return Vec2Half(_data.x * b, _data.y * b);
    }

    CPT_GPU_INLINE
    Vec2Half &operator*=(half b) noexcept {
        _data.x *= b;
        _data.y *= b;
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE
    Vec2Half operator*(VecType &&b) const noexcept {
        return Vec2Half(_data.x * b.x(), _data.y * b.y());
    }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE
    Vec2Half &operator*=(VecType &&b) noexcept {
        _data.x *= b.x();
        _data.y *= b.y();
        return *this;
    }

    CPT_CPU_GPU_INLINE operator half2() const { return _data; }
    CPT_CPU_GPU_INLINE operator Vec2() const { return Vec2(xy_float()); }
};

/**
 * This data type is used in storing the UV coords
 * of a primitive in device memory (SoA3), compared to float SoA3
 * We only need 4 float (compared with 6) therefore
 * can be more memory efficient
 */
class PackedHalf2 {
  private:
    uint4 data; // half4
  public:
    CPT_GPU
    PackedHalf2() {}

    CONDITION_TEMPLATE_SEP_3(VT1, VT2, VT3, Vec2, Vec2, Vec2)
    CPT_CPU_GPU
    PackedHalf2(VT1 &&v1, VT2 &&v2, VT3 &&v3) {
        HALF2(data.x) = Vec2Half(v1.x(), v1.y());
        HALF2(data.y) = Vec2Half(v2.x(), v2.y());
        HALF2(data.z) = Vec2Half(v3.x(), v3.y());
    }

    CPT_GPU_INLINE Vec2Half x() const { return CONST_HALF2(data.x); }
    CPT_GPU_INLINE Vec2Half y() const { return CONST_HALF2(data.y); }
    CPT_GPU_INLINE Vec2Half z() const { return CONST_HALF2(data.z); }

    CPT_GPU_INLINE Vec2Half lerp(float u, float v) const {
        Vec2Half h1 = CONST_HALF2(data.x), h2 = CONST_HALF2(data.y),
                 h3 = CONST_HALF2(data.z);
        return h1 * (1.f - u - v) + h2 * u + h3 * v;
    }
};

CONDITION_TEMPLATE(VecType, Vec2Half)
CPT_GPU_INLINE void print_vec2_half(VecType &&obj) {
    auto temp = obj.xy_float();
    printf("[%f, %f]\n", temp.x, temp.y);
}
