/**
 * @brief CUDA/CPU 2D vector implementation
 * @author: Qianyue He
 * @date:   4.10.2024
*/
#pragma once
#include "core/cuda_utils.cuh"
#include "vec3.cuh"

class Vec2 {
private:
    float2 _data;
public:
    CPT_CPU_GPU
    Vec2(float _x = 0, float _y = 0): 
        _data(make_float2(_x, _y)) {}

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

    CPT_CPU_GPU const float& x() const { return _data.x; }
    CPT_CPU_GPU const float& y() const { return _data.y; }

    CPT_CPU_GPU
    Vec2 abs() const noexcept {
        return Vec2(fabs(_data.x), fabs(_data.y));
    }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU
    Vec2 operator+(VecType&& b) const noexcept { return Vec2(_data.x + b.x(), _data.y + b.y()); }

    CONDITION_TEMPLATE(VecType, Vec2)
    void operator+=(VecType&& b) noexcept {
        _data.x += b.x();
        _data.y += b.y();
    }

    CPT_CPU_GPU
    Vec2 operator-() const noexcept {
        return Vec2(-_data.x, -_data.y);
    }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU
    void operator-=(VecType&& b) noexcept {
        _data.x -= b.x();
        _data.y -= b.y();
    }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU
    Vec2 operator-(VecType&& b) const { return Vec2(_data.x - b.x(), _data.y - b.y()); }

    CPT_CPU_GPU
    Vec2 operator*(float b) const noexcept { return Vec2(_data.x * b, _data.y * b); }

    CPT_CPU_GPU
    void operator*=(float b) noexcept {
        _data.x *= b;
        _data.y *= b;
    }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU
    Vec2 operator*(VecType&& b) const noexcept { return Vec2(_data.x * b.x(), _data.y * b.y(),); }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU
    void operator*=(VecType&& b) noexcept {
        _data.x *= b.x();
        _data.y *= b.y();
    }


    CPT_CPU_GPU
    Vec2 normalized() const { return *this * (1.f / length()); }

    CPT_CPU_GPU
    void normalize() { this->operator*=(1.f / length()); }

    CPT_CPU_GPU
    float length2() const { return _data.x * _data.x + _data.y * _data.y; }

    CPT_CPU_GPU
    float length() const { return sqrt(length2()); }

    CONDITION_TEMPLATE(VecType, Vec2)
    CPT_CPU_GPU
    float dot(VecType&& b) const noexcept { return _data.x * b.x() + _data.y * b.y(); }

    CPT_CPU_GPU
    float max_elem() const noexcept { return max(_data.x, _data.y); }

    CPT_CPU_GPU
    float min_elem() const noexcept { return min(_data.x, _data.y); }
};

CPT_CPU_GPU void print_vec2(const Vec2& obj) {
    printf("[%f, %f]\n", obj.x(), obj.y());
}