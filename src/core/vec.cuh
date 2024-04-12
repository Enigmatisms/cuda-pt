#pragma once
#include <iostream>
#include <chrono>
#include <iomanip>
#include <curand_kernel.h>

#include "cuda_utils.cuh"

template <typename Ty>
struct Vec3 {
    Ty x, y, z;

    CPT_CPU_GPU
    Vec3() : x(0), y(0), z(0) {}

    CPT_CPU_GPU
    Vec3(Ty _x, Ty _y, Ty _z): x(_x), y(_y), z(_z) {}

    CPT_CPU_GPU 
    float& operator[](int index) {
        return *((&x) + index);
    }

    CPT_CPU_GPU 
    Ty operator[](int index) const {
        return *((&x) + index);
    }

    CPT_CPU_GPU
    Vec3 abs() const {
        return Vec3(::abs(x), ::abs(y), ::abs(z));
    }

    CPT_CPU_GPU
    Vec3 operator+(const Vec3 &b) const { return Vec3(x + b.x, y + b.y, z + b.z); }

    CPT_CPU_GPU
    void operator+=(const Vec3 &b) {
        x += b.x;
        y += b.y;
        z += b.z;
    }

    CPT_CPU_GPU
    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    CPT_CPU_GPU
    Vec3 operator-(const Vec3 &b) const { return Vec3(x - b.x, y - b.y, z - b.z); }

    CPT_CPU_GPU
    Vec3 operator*(Ty b) const { return Vec3(x * b, y * b, z * b); }

    CPT_CPU_GPU
    void operator*=(Ty b) {
        x *= b;
        y *= b;
        z *= b;
    }

    CPT_CPU_GPU
    Vec3 operator*(const Vec3 &b) const { return Vec3(x * b.x, y * b.y, z * b.z); }

    CPT_CPU_GPU
    void operator*=(const Vec3 &b) {
        x *= b.x;
        y *= b.y;
        z *= b.z;
    }

    CPT_CPU_GPU
    Vec3 normalized() const { return *this * (1.f / length()); }

    CPT_CPU_GPU
    void normalize() { this->operator*=(1.f / length()); }

    CPT_CPU_GPU
    Ty length2() const { return x * x + y * y + z * z; }

    CPT_CPU_GPU
    Ty length() const { return sqrt(length2()); }

    CPT_CPU_GPU
    Ty dot(const Vec3 &b) const { return x * b.x + y * b.y + z * b.z; }

    CPT_CPU_GPU
    Ty dot(Vec3&& b) const { return x * b.x + y * b.y + z * b.z; }

    CPT_CPU_GPU
    Vec3 cross(const Vec3 &b) const {
        return Vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
    }

    CPT_CPU_GPU
    Vec3 cross(Vec3&& b) const {
        return Vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
    }

    CPT_CPU_GPU
    Ty max_elem() const { return max(x, max(y, z)); }

    CPT_CPU_GPU
    Ty min_elem() const { return min(x, min(y, z)); }
};

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;