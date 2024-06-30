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
    int ray_tag;

    // ray_tag: the highest 4 bits:
    // 31, 30, 29: reserved, 28: active, if 0: inactive, if 1: active

    CONDITION_TEMPLATE_2(T1, T2, Vec3)
    CPT_CPU_GPU
    Ray(T1&& o_, T2&& d_, float hitT = MAX_DIST) : o(std::forward<T1>(o_)), hit_t(hitT), d(std::forward<T2>(d_)), ray_tag(0) {}

    bool is_active() const noexcept {
        return (ray_tag >> 28) & 1;
    }

    int hit_id() const noexcept {
        return ray_tag & 0x0fffffff;
    }

    void set_hit_index(int min_index) noexcept {
        ray_tag = (0xf0000000 & ray_tag) | min_index; 
    }

    void set_hit_status(bool hit) noexcept {
        ray_tag &= 0xefffffff;      // clear bit 28
        ray_tag |= hit << 28;       // set bit 28
    }
};
