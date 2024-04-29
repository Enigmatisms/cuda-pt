/**
 * @brief CUDA/CPU 3D vector implementation
 * @author: Qianyue He
 * @date:   4.10.2024
*/
#pragma once
#include <iostream>
#include <chrono>
#include <iomanip>
#include "cuda_utils.cuh"

template <typename Ty>
struct Vec3 {
    Ty x, y, z;

    CPT_CPU_GPU
    Vec3() : x(0), y(0), z(0) {}

    template <typename T1, typename T2, typename T3>
    CPT_CPU_GPU
    Vec3(T1 _x, T2 _y, T3 _z): 
        x(static_cast<Ty>(_x)), 
        y(static_cast<Ty>(_y)), 
        z(static_cast<Ty>(_z)) {}

    CPT_CPU_GPU 
    Ty& operator[](int index) {
        return *((&x) + index);
    }

    CPT_CPU_GPU 
    Ty operator[](int index) const {
        return *((&x) + index);
    }

    CPT_CPU_GPU
    Vec3<Ty> abs() const noexcept {
        return Vec3<Ty>(::abs(x), ::abs(y), ::abs(z));
    }

    CONDITION_TEMPLATE(VecType, Vec3<Ty>)
    CPT_CPU_GPU
    Vec3<Ty> operator+(VecType&& b) const noexcept { return Vec3<Ty>(x + b.x, y + b.y, z + b.z); }

    CONDITION_TEMPLATE(VecType, Vec3<Ty>)
    CPT_CPU_GPU
    void operator+=(VecType&& b) noexcept {
        x += b.x;
        y += b.y;
        z += b.z;
    }

    CPT_CPU_GPU
    Vec3<Ty> operator-() const noexcept {
        return Vec3<Ty>(-x, -y, -z);
    }

    CONDITION_TEMPLATE(VecType, Vec3<Ty>)
    CPT_CPU_GPU
    void operator-=(VecType&& b) noexcept {
        x -= b.x;
        y -= b.y;
        z -= b.z;
    }

    CONDITION_TEMPLATE(VecType, Vec3<Ty>)
    CPT_CPU_GPU
    Vec3<Ty> operator-(VecType&& b) const { return Vec3<Ty>(x - b.x, y - b.y, z - b.z); }

    CPT_CPU_GPU
    Vec3<Ty> operator*(Ty b) const noexcept { return Vec3<Ty>(x * b, y * b, z * b); }

    CPT_CPU_GPU
    void operator*=(Ty b) noexcept {
        x *= b;
        y *= b;
        z *= b;
    }

    CONDITION_TEMPLATE(VecType, Vec3<Ty>)
    CPT_CPU_GPU
    Vec3<Ty> operator*(VecType&& b) const noexcept { return Vec3<Ty>(x * b.x, y * b.y, z * b.z); }

    CONDITION_TEMPLATE(VecType, Vec3<Ty>)
    CPT_CPU_GPU
    void operator*=(VecType&& b) noexcept {
        x *= b.x;
        y *= b.y;
        z *= b.z;
    }

    CPT_CPU_GPU
    Vec3<Ty> normalized() const { return *this * (1.f / length()); }

    CPT_CPU_GPU
    void normalize() { this->operator*=(1.f / length()); }

    CPT_CPU_GPU
    Ty length2() const { return x * x + y * y + z * z; }

    CPT_CPU_GPU
    Ty length() const { return sqrt(length2()); }

    CONDITION_TEMPLATE(VecType, Vec3<Ty>)
    CPT_CPU_GPU
    Ty dot(VecType&& b) const noexcept { return x * b.x + y * b.y + z * b.z; }

    CONDITION_TEMPLATE(VecType, Vec3<Ty>)
    CPT_CPU_GPU
    Vec3<Ty> cross(VecType&& b) const noexcept {
        return Vec3<Ty>(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
    }

    CPT_CPU_GPU
    Ty max_elem() const noexcept { return max(x, max(y, z)); }

    CPT_CPU_GPU
    Ty min_elem() const noexcept { return min(x, min(y, z)); }
};

template <typename Ty>
CPT_CPU_GPU void print_vec3(const Vec3<Ty>& obj) {
    printf("[%f, %f, %f]\n", obj.x, obj.y, obj.z);
}

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;