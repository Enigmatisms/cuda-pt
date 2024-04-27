/**
 * @brief CUDA/CPU 2D vector implementation
 * @author: Qianyue He
 * @date:   4.10.2024
*/
#pragma once
#include <iostream>
#include <chrono>
#include <iomanip>
#include <curand_kernel.h>
#include "cuda_utils.cuh"
#include "vec3.cuh"

template <typename Ty>
struct Vec2 {
    Ty x, y;

    CPT_CPU_GPU
    Vec2() : x(0), y(0) {}

    CPT_CPU_GPU
    Vec2(Ty _x, Ty _y, Ty _z): x(_x), y(_y) {}

    CPT_CPU_GPU 
    Ty& operator[](int index) {
        return *((&x) + index);
    }

    CPT_CPU_GPU 
    Ty operator[](int index) const {
        return *((&x) + index);
    }

    CPT_CPU_GPU
    Vec2<Ty> abs() const noexcept {
        return Vec2<Ty>(::abs(x), ::abs(y));
    }

    CONDITION_TEMPLATE(VecType, Vec2<Ty>)
    CPT_CPU_GPU
    Vec2<Ty> operator+(VecType&& b) const noexcept { return Vec2<Ty>(x + b.x, y + b.y); }

    CONDITION_TEMPLATE(VecType, Vec2<Ty>)
    CPT_CPU_GPU
    void operator+=(VecType&& b) noexcept {
        x += b.x;
        y += b.y;
    }

    CPT_CPU_GPU
    Vec2<Ty> operator-() const noexcept {
        return Vec2<Ty>(-x, -y);
    }

    CONDITION_TEMPLATE(VecType, Vec2<Ty>)
    CPT_CPU_GPU
    void operator-=(VecType&& b) noexcept {
        x -= b.x;
        y -= b.y;
    }

    CONDITION_TEMPLATE(VecType, Vec2<Ty>)
    CPT_CPU_GPU
    Vec2<Ty> operator-(VecType&& b) const { return Vec2<Ty>(x - b.x, y - b.y); }

    CPT_CPU_GPU
    Vec2<Ty> operator*(Ty b) const noexcept { return Vec2<Ty>(x * b, y * b); }

    CPT_CPU_GPU
    void operator*=(Ty b) noexcept {
        x *= b;
        y *= b;
    }

    CONDITION_TEMPLATE(VecType, Vec2<Ty>)
    CPT_CPU_GPU
    Vec2<Ty> operator*(VecType&& b) const noexcept { return Vec2<Ty>(x * b.x, y * b.y); }

    CONDITION_TEMPLATE(VecType, Vec2<Ty>)
    CPT_CPU_GPU
    void operator*=(VecType&& b) noexcept {
        x *= b.x;
        y *= b.y;
    }

    CPT_CPU_GPU
    Vec3<Ty> expand(Ty z = 1) const noexcept { return Vec3<Ty>(x, y, z); }

    CPT_CPU_GPU
    Vec2<Ty> normalized() const { return *this * (1.f / length()); }

    CPT_CPU_GPU
    void normalize() { this->operator*=(1.f / length()); }

    CPT_CPU_GPU
    Ty length2() const { return x * x + y * y; }

    CPT_CPU_GPU
    Ty length() const { return sqrt(length2()); }

    CONDITION_TEMPLATE(VecType, Vec2<Ty>)
    CPT_CPU_GPU
    Ty dot(VecType&& b) const noexcept { return x * b.x + y * b.y; }

    CPT_CPU_GPU
    Ty max_elem() const noexcept { return max(x, y); }

    CPT_CPU_GPU
    Ty min_elem() const noexcept { return min(x, y); }
};

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;