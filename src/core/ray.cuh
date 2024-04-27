#pragma once
#include "vec3.cuh"

template <typename Ty>
struct RayBase {
    Vec3<Ty> o, d;

    CONDITION_TEMPLATE_2(T1, T2, Vec3<Ty>)
    CPT_CPU_GPU
    RayBase(T1&& o_, T2&& d_) : o(std::forward<T1>(o_)), d(std::forward<T2>(d_)) {}
};

using Ray  = RayBase<float>;
using Rayd = RayBase<double>;