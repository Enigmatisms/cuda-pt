/**
 * @brief CUDA/CPU 2D vector (Half) implementation
 * Note that Half Vector is only used in GPU, host has no access to half
 * @author: Qianyue He
 * @date:  2024.11.4
*/
#pragma once
#include <cuda_fp16.h>
#include "core/cuda_utils.cuh"
#include "vec2.cuh"

#define HALF2(v) (*(reinterpret_cast<half2*>(&v)))
#define CONST_HALF2(v) (*(reinterpret_cast<const half2*>(&v)))

class Vec2Half {
private:
    half2 _data;
public:
    CPT_CPU_GPU Vec2Half() {}

    CPT_CPU_GPU
    Vec2Half(float _v): 
        _data(make_half2(
            __float2half(_v), 
            __float2half(_v)
        )
    ) {}

    CONDITION_TEMPLATE_DEFAULT(Half2Type, half2)
    CPT_CPU_GPU
    Vec2Half(Half2Type&& _h): 
        _data(std::forward<Half2Type>(_h)) {}

    CONDITION_TEMPLATE_DEFAULT(Vec2Type, Vec2)
    CPT_GPU
    Vec2Half(Vec2Type&& _h): 
        _data(_h.x(), _h.y()) {}

    CPT_CPU_GPU
    Vec2Half(float _x, float _y): 
        _data(make_half2(
            __float2half(_x), 
            __float2half(_y)
        )
    ) {}

    CPT_GPU
    Vec2Half(half _v): 
        _data(make_half2(_v, _v)) {}

    CPT_GPU
    Vec2Half(half _x, half _y): 
        _data(make_half2(_x, _y)) {}

    CPT_GPU_INLINE 
    half& operator[](int index) {
        return *((&_data.x) + index);
    }

    CPT_GPU_INLINE 
    half operator[](int index) const {
        return *((&_data.x) + index);
    }

    CPT_CPU_GPU_INLINE half& x() { return _data.x; }
    CPT_CPU_GPU_INLINE half& y() { return _data.y; }

    CPT_CPU_GPU_INLINE float x_float() const noexcept { return __half2float(_data.x); }
    CPT_CPU_GPU_INLINE float y_float() const noexcept { return __half2float(_data.y); }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE Vec2Half operator+(VecType&& b) const noexcept { return Vec2Half(_data.x + b.x(), _data.y + b.y()); }

    CPT_GPU_INLINE Vec2Half operator+(float b) const noexcept { 
        auto v = __float2half(b);
        return Vec2Half(_data.x + v, _data.y + v); 
    }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    Vec2Half& operator+=(VecType&& b) noexcept {
        _data.x += b.x();
        _data.y += b.y();
        return *this;
    }

    CPT_GPU_INLINE
    Vec2Half operator-() const noexcept {
        return Vec2Half(-_data.x, -_data.y);
    }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE
    Vec2Half& operator-=(VecType&& b) noexcept {
        _data.x -= b.x();
        _data.y -= b.y();
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE
    Vec2Half operator-(VecType&& b) const { return Vec2Half(_data.x - b.x(), _data.y - b.y()); }

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
    Vec2Half& operator*=(float b) noexcept {
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
    Vec2Half& operator*=(half b) noexcept {
        _data.x *= b;
        _data.y *= b;
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE
    Vec2Half operator*(VecType&& b) const noexcept { return Vec2Half(_data.x * b.x(), _data.y * b.y()); }

    CONDITION_TEMPLATE(VecType, Vec2Half)
    CPT_GPU_INLINE
    Vec2Half& operator*=(VecType&& b) noexcept {
        _data.x *= b.x();
        _data.y *= b.y();
        return *this;
    }

    CPT_CPU_GPU_INLINE operator half2() const { return _data; }
    CPT_CPU_GPU_INLINE operator Vec2() const {
        return Vec2(x_float(), y_float());
    }
};

/**
 * This data type is used in storing the UV coords
 * of a primitive in device memory (SoA3), compared to float SoA3
 * We only need 4 float (compared with 6) therefore
 * can be more memory efficient
 */
class PackedHalf2 {
private:
    uint4 data;     // half4
public:
    CPT_GPU
    PackedHalf2() {}

    CONDITION_TEMPLATE_SEP_3(VT1, VT2, VT3, Vec2, Vec2, Vec2)
    CPT_CPU_GPU
    PackedHalf2(VT1&& v1, VT2&& v2, VT3&& v3) {
        HALF2(data.x) = Vec2Half(v1.x(), v1.y());
        HALF2(data.y) = Vec2Half(v2.x(), v2.y());
        HALF2(data.z) = Vec2Half(v3.x(), v3.y());
    }

    CPT_GPU_INLINE Vec2Half x() const { return CONST_HALF2(data.x); }
    CPT_GPU_INLINE Vec2Half y() const { return CONST_HALF2(data.y); }
    CPT_GPU_INLINE Vec2Half z() const { return CONST_HALF2(data.z); }

    CPT_GPU_INLINE Vec2Half lerp(float u, float v) const {
        Vec2Half h1 = CONST_HALF2(data.x), 
                 h2 = CONST_HALF2(data.y),
                 h3 = CONST_HALF2(data.z);
        return h1 * (1.f - u - v) + h2 * u + h3 * v;
    }
};

CPT_GPU_INLINE void print_vec2_half(const Vec2Half& obj) {
    printf("[%f, %f]\n", obj.x_float(), obj.y_float());
}