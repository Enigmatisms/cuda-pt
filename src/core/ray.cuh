#pragma once
#include "vec3.cuh"

struct Ray {
    Vec3 o, d;

    CONDITION_TEMPLATE_2(T1, T2, Vec3)
    CPT_CPU_GPU
    Ray(T1&& o_, T2&& d_) : o(std::forward<T1>(o_)), d(std::forward<T2>(d_)) {}
};