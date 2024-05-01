/**
 * @brief CUDA/CPU 3D vector implementation
 * @author: Qianyue He
 * @date:   4.10.2024
*/
#pragma once
#include <iostream>
#include <chrono>
#include <iomanip>
#include "core/cuda_utils.cuh"

class Vec3 {
private:
    float3 _data;
public:
    CPT_CPU_GPU
    Vec3(float _x = 0, float _y = 0, float _z = 0): 
        _data(make_float3(_x, _y, _z)) {}

    CPT_CPU_GPU 
    float& operator[](int index) {
        return *((&_data.x) + index);
    }

    CPT_CPU_GPU 
    float operator[](int index) const {
        return *((&_data.x) + index);
    }

    CPT_CPU_GPU float& x() { return _data.x; }
    CPT_CPU_GPU float& y() { return _data.y; }
    CPT_CPU_GPU float& z() { return _data.z; }

    CPT_CPU_GPU const float& x() const { return _data.x; }
    CPT_CPU_GPU const float& y() const { return _data.y; }
    CPT_CPU_GPU const float& z() const { return _data.z; }

    CPT_CPU_GPU
    Vec3 abs() const noexcept {
        return Vec3(fabs(_data.x), fabs(_data.y), fabs(_data.z));
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU
    Vec3 operator+(VecType&& b) const noexcept { return Vec3(_data.x + b.x(), _data.y + b.y(), _data.z + b.z()); }

    CONDITION_TEMPLATE(VecType, Vec3)
    void operator+=(VecType&& b) noexcept {
        _data.x += b.x();
        _data.y += b.y();
        _data.z += b.z();
    }

    CPT_CPU_GPU
    Vec3 operator-() const noexcept {
        return Vec3(-_data.x, -_data.y, -_data.z);
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU
    void operator-=(VecType&& b) noexcept {
        _data.x -= b.x();
        _data.y -= b.y();
        _data.z -= b.z();
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU
    Vec3 operator-(VecType&& b) const { return Vec3(_data.x - b.x(), _data.y - b.y(), _data.z - b.z()); }

    CPT_CPU_GPU
    Vec3 operator*(float b) const noexcept { return Vec3(_data.x * b, _data.y * b, _data.z * b); }

    CPT_CPU_GPU
    void operator*=(float b) noexcept {
        _data.x *= b;
        _data.y *= b;
        _data.z *= b;
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU
    Vec3 operator*(VecType&& b) const noexcept { return Vec3(_data.x * b.x(), _data.y * b.y(), _data.z * b.z()); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU
    void operator*=(VecType&& b) noexcept {
        _data.x *= b.x();
        _data.y *= b.y();
        _data.z *= b.z();
    }


    CPT_CPU_GPU
    Vec3 normalized() const { return *this * (1.f / length()); }

    CPT_CPU_GPU
    void normalize() { this->operator*=(1.f / length()); }

    CPT_CPU_GPU
    float length2() const { return _data.x * _data.x + _data.y * _data.y + _data.z * _data.z; }

    CPT_CPU_GPU
    float length() const { return sqrt(length2()); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU
    float dot(VecType&& b) const noexcept { return _data.x * b.x() + _data.y * b.y() + _data.z * b.z(); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU
    Vec3 cross(VecType&& b) const noexcept {
        return Vec3(_data.y * b.z() - _data.z * b.y(), _data.z * b.x() - _data.x * b.z(), _data.x * b.y() - _data.y * b.x());
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU
    float maximize(VecType&& b) const noexcept { return Vec3(max(_data.x, b.x()), max(_data.y, b.y()), max(_data.z, b.z())); }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU_GPU
    float minimize(VecType&& b) const noexcept { return Vec3(min(_data.x, b.x()), min(_data.y, b.y()), min(_data.z, b.z())); }

    CPT_CPU_GPU
    float max_elem() const noexcept { return max(_data.x, max(_data.y, _data.z)); }

    CPT_CPU_GPU
    float min_elem() const noexcept { return min(_data.x, min(_data.y, _data.z)); }
};

CPT_CPU_GPU void print_vec3(const Vec3& obj) {
    printf("[%f, %f, %f]\n", obj.x(), obj.y(), obj.z());
}

CONDITION_TEMPLATE(VecType, Vec3)
CPT_CPU_GPU
Vec3 operator*(float b, VecType&& v) noexcept { return Vec3(v.x() * b, v.y() * b, v.z() * b); }

CONDITION_TEMPLATE(VecType, Vec3)
CPT_CPU_GPU
Vec3 operator/(float b, VecType&& v) noexcept { return Vec3(b / v.x(), b / v.y(), b / v.z()); }