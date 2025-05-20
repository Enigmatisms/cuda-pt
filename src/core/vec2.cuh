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
 * @brief CUDA/CPU 2D vector implementation
 * @date:   2024.4.10
 */
#pragma once
#include "core/cuda_utils.cuh"
#include "vec3.cuh"

class Vec2 {
  private:
    float2 _data;

  public:
    CPT_CPU_GPU Vec2() {}

    CPT_CPU_GPU
    constexpr Vec2(float _v) : _data({_v, _v}) {}

    CPT_CPU_GPU
    constexpr Vec2(float _x, float _y) : _data({_x, _y}) {}

    CPT_CPU_GPU
    Vec2(float2 &&v) : _data(std::move(v)) {}

    CPT_CPU_GPU_INLINE
    float &operator[](int index) { return *((&_data.x) + index); }

    CPT_CPU_GPU_INLINE
    float operator[](int index) const { return *((&_data.x) + index); }

    CPT_CPU_GPU_INLINE float &x() { return _data.x; }
    CPT_CPU_GPU_INLINE float &y() { return _data.y; }

    constexpr CPT_CPU_GPU_INLINE const float &x() const { return _data.x; }
    constexpr CPT_CPU_GPU_INLINE const float &y() const { return _data.y; }

    CPT_CPU_GPU_INLINE
    Vec2 abs() const noexcept { return Vec2(fabs(_data.x), fabs(_data.y)); }

    CONDITION_TEMPLATE(VecType, Vec2)
    constexpr CPT_CPU_GPU_INLINE Vec2 operator+(VecType &&b) const noexcept {
        return Vec2(_data.x + b.x(), _data.y + b.y());
    }

    constexpr CPT_CPU_GPU_INLINE Vec2 operator+(float b) const noexcept {
        return Vec2(_data.x + b, _data.y + b);
    }

    CONDITION_TEMPLATE(VecType, Vec2)
    Vec2 &operator+=(VecType &&b) noexcept {
        _data.x += b.x();
        _data.y += b.y();
        return *this;
    }

    CPT_CPU_GPU_INLINE
    constexpr Vec2 operator-() const noexcept {
        return Vec2(-_data.x, -_data.y);
    }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU_INLINE
    Vec2 &operator-=(VecType &&b) noexcept {
        _data.x -= b.x();
        _data.y -= b.y();
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU_INLINE
    constexpr Vec2 operator-(VecType &&b) const {
        return Vec2(_data.x - b.x(), _data.y - b.y());
    }

    CPT_CPU_GPU_INLINE Vec2 operator-(float b) const {
        return Vec2(_data.x - b, _data.y - b);
    }

    CPT_CPU_GPU_INLINE
    constexpr Vec2 operator*(float b) const noexcept {
        return Vec2(_data.x * b, _data.y * b);
    }

    CPT_CPU_GPU_INLINE
    Vec2 &operator*=(float b) noexcept {
        _data.x *= b;
        _data.y *= b;
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU_INLINE
    constexpr Vec2 operator*(VecType &&b) const noexcept {
        return Vec2(_data.x * b.x(), _data.y * b.y());
    }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU_INLINE
    Vec2 &operator*=(VecType &&b) noexcept {
        _data.x *= b.x();
        _data.y *= b.y();
        return *this;
    }

    CPT_GPU_INLINE
    Vec2 normalized() const { return *this * rhypotf(_data.x, _data.y); }

    CPT_GPU_INLINE
    void normalize() { this->operator*=(rhypotf(_data.x, _data.y)); }

    CPT_CPU_GPU_INLINE
    float length2() const { return _data.x * _data.x + _data.y * _data.y; }

    CPT_GPU_INLINE
    float length() const { return hypotf(_data.x, _data.y); }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU_INLINE
    float dot(VecType &&b) const noexcept {
        return _data.x * b.x() + _data.y * b.y();
    }

    CPT_CPU_GPU_INLINE
    float max_elem() const noexcept { return max(_data.x, _data.y); }

    CPT_CPU_GPU_INLINE
    float min_elem() const noexcept { return min(_data.x, _data.y); }

    CPT_CPU_GPU_INLINE operator float2() const { return _data; }
};

CPT_CPU_GPU_INLINE void print_vec2(const Vec2 &obj) {
    printf("[%f, %f]\n", obj.x(), obj.y());
}
