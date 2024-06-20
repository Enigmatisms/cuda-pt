/**
 * Ray definition
 * TODO (6.18): Stream Compaction
 * @author: Qianyue He
 * @date:  (modified) 6.18.2024
*/

#pragma once
#include "vec3.cuh"

// the ray contains 32 Bytes (128 bit)
struct Ray {
    Vec3 o;
    float hit_t;
    Vec3 d;
    uint32_t ray_tag;

    // we might only use eight of the bits
    // bit 0: is inactive, 1 means inactive

    CONDITION_TEMPLATE_2(T1, T2, Vec3)
    CPT_CPU_GPU
    Ray(T1&& o_, T2&& d_, float hitT = MAX_DIST) : o(std::forward<T1>(o_)), hit_t(hitT), d(std::forward<T2>(d_)), ray_tag(0) {}
};
