/**
 * @brief CUDA/CPU 3D vector implementation
 * @author: Qianyue He
 * @date:   4.10.2024
*/
#pragma once
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cuda_runtime_api.h>
#include "core/cuda_utils.cuh"
#include "core/constants.cuh"

class Vec3 {
private:
    float3 _data;
public:
    CPT_CPU_GPU Vec3() {}
    CPT_CPU_GPU
    Vec3(float _x, float _y, float _z): 
        _data(make_float3(_x, _y, _z)) {}

    CPT_CPU_GPU_INLINE 
    float& operator[](int index) {
        return *((&_data.x) + index);
    }

    CPT_CPU_GPU_INLINE 
    float operator[](int index) const {
        return *((&_data.x) + index);
    }

    CPT_CPU_GPU_INLINE float& x() { return _data.x; }
    CPT_CPU_GPU_INLINE float& y() { return _data.y; }
    CPT_CPU_GPU_INLINE float& z() { return _data.z; }

    CPT_CPU_GPU_INLINE float x() const { return _data.x; }
    CPT_CPU_GPU_INLINE float y() const { return _data.y; }
    CPT_CPU_GPU_INLINE float z() const { return _data.z; }

    CPT_CPU_GPU_INLINE
    Vec3 abs() const noexcept {
        return Vec3(fabs(_data.x), fabs(_data.y), fabs(_data.z));
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 operator+(VecType&& b) const noexcept { return Vec3(_data.x + b.x(), _data.y + b.y(), _data.z + b.z()); }

    CPT_CPU_GPU_INLINE
    Vec3 operator+(float b) const noexcept { return Vec3(_data.x + b, _data.y + b, _data.z + b); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3& operator+=(VecType&& b) noexcept {
        _data.x += b.x();
        _data.y += b.y();
        _data.z += b.z();
        return *this;
    }

    CPT_CPU_GPU_INLINE
    Vec3& operator+=(float v) noexcept {
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
    Vec3 operator-(float b) const noexcept { return Vec3(_data.x - b, _data.y - b, _data.z - b); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3& operator-=(VecType&& b) noexcept {
        _data.x -= b.x();
        _data.y -= b.y();
        _data.z -= b.z();
        return *this;
    }

    CPT_CPU_GPU_INLINE
    Vec3& operator-=(float v) noexcept {
        _data.x -= v;
        _data.y -= v;
        _data.z -= v;
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 operator-(VecType&& b) const { return Vec3(_data.x - b.x(), _data.y - b.y(), _data.z - b.z()); }

    CPT_CPU_GPU_INLINE
    Vec3 operator*(float b) const noexcept { return Vec3(_data.x * b, _data.y * b, _data.z * b); }

    CPT_CPU_GPU_INLINE
    Vec3& operator*=(float b) noexcept {
        _data.x *= b;
        _data.y *= b;
        _data.z *= b;
        return *this;
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 operator*(VecType&& b) const noexcept { return Vec3(_data.x * b.x(), _data.y * b.y(), _data.z * b.z()); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    void operator*=(VecType&& b) noexcept {
        _data.x *= b.x();
        _data.y *= b.y();
        _data.z *= b.z();
    }

    CPT_CPU_GPU_INLINE void fill(float v) noexcept {
        _data = make_float3(v, v, v);
    }

    CPT_CPU_GPU_INLINE
    bool numeric_err() const noexcept {
        return isnan(_data.x) || isnan(_data.y) || isnan(_data.z) || \
               isinf(_data.x) || isinf(_data.y) || isinf(_data.z);
    }

    CPT_CPU_GPU_INLINE
    Vec3 normalized() const { return *this * (1.f / length()); }

    CPT_CPU_GPU_INLINE
    void normalize() { this->operator*=(1.f / length()); }

    CPT_CPU_GPU_INLINE
    float length2() const { return _data.x * _data.x + _data.y * _data.y + _data.z * _data.z; }

    CPT_CPU_GPU_INLINE
    float length() const { return sqrt(length2()); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    float dot(VecType&& b) const noexcept { return _data.x * b.x() + _data.y * b.y() + _data.z * b.z(); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 cross(VecType&& b) const noexcept {
        return Vec3(_data.y * b.z() - _data.z * b.y(), _data.z * b.x() - _data.x * b.z(), _data.x * b.y() - _data.y * b.x());
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 maximize(VecType&& b) const noexcept { return Vec3(max(_data.x, b.x()), max(_data.y, b.y()), max(_data.z, b.z())); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU_INLINE
    Vec3 minimize(VecType&& b) const noexcept { return Vec3(min(_data.x, b.x()), min(_data.y, b.y()), min(_data.z, b.z())); }

    CPT_CPU_GPU_INLINE
    float max_elem() const noexcept { return max(_data.x, max(_data.y, _data.z)); }

    CPT_CPU_GPU_INLINE
    float min_elem() const noexcept { return min(_data.x, min(_data.y, _data.z)); }
};

CPT_CPU_GPU void print_vec3(const Vec3& obj) {
    printf("[%f, %f, %f]\n", obj.x(), obj.y(), obj.z());
}

CONDITION_TEMPLATE(VecType, Vec3)
CPT_CPU_GPU_INLINE
Vec3 operator*(float b, VecType&& v) noexcept { return Vec3(v.x() * b, v.y() * b, v.z() * b); }

CONDITION_TEMPLATE(VecType, Vec3)
CPT_CPU_GPU_INLINE
Vec3 operator/(float b, VecType&& v) noexcept { return Vec3(b / v.x(), b / v.y(), b / v.z()); }