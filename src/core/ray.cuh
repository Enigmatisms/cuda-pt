/**
 * Ray definition
 * TODO (6.18): Stream Compaction
 * @author: Qianyue He
 * @date:  (modified) 6.18.2024
*/

#pragma once
#include "vec3.cuh"

// the ray contains 32 Bytes (128 bit). Our ray is able to be transfered
// by two LDS.128 / ST.128
struct Ray {
    Vec3 o;
    float hit_t;
    Vec3 d;
    uint32_t ray_tag;

    CPT_CPU_GPU Ray() {}

    // ray_tag: the highest 4 bits:
    // 28: active, if 0: inactive, if 1: active
    // 29: hit, if 0: not hit (potentially inactive), 1 hit
    // 30: the last hit is delta? 1: is delta (for MIS use), 0: non-delta
    // 31: reserved, not used
    // the ray is marked active since construction
    CONDITION_TEMPLATE_2(T1, T2, Vec3)
    CPT_CPU_GPU
    Ray(T1&& o_, T2&& d_, float hitT = MAX_DIST) : 
        o(std::forward<T1>(o_)), hit_t(hitT), d(std::forward<T2>(d_)), ray_tag(0x10000000) {}

    CPT_CPU_GPU_INLINE
    Vec3 advance(float t) const noexcept {
        return Vec3(fmaf(d.x(), t, o.x()), fmaf(d.y(), t, o.y()), fmaf(d.z(), t, o.z()));
    } 

    CPT_CPU_GPU_INLINE bool is_active() const noexcept {
        return (ray_tag & 0x10000000) > 0;
    }

    CPT_CPU_GPU_INLINE void set_active(bool v) noexcept {
        ray_tag &= 0xefffffff;      // clear bit 28
        ray_tag |= uint32_t(v) << 28;
    }

    CPT_CPU_GPU_INLINE uint32_t hit_id() const noexcept {
        return ray_tag & 0x0fffffff;
    }

    CPT_CPU_GPU_INLINE void set_hit_index(uint32_t min_index) noexcept {
        ray_tag = (0xf0000000 & ray_tag) | min_index; 
    }

    CPT_CPU_GPU_INLINE bool is_hit() const noexcept {
        return (ray_tag & 0x20000000) > 0;
    }

    CPT_CPU_GPU_INLINE void clr_hit() noexcept {
        ray_tag &= 0xdfffffff;      // clear bit 29
    }

    CPT_CPU_GPU_INLINE void set_hit(bool is_hit = true) noexcept {
        // set bit 29 or clear bit 29
        ray_tag = is_hit ? ray_tag | (1 << 29) : ray_tag & 0xdfffffff;
    }

    CPT_CPU_GPU_INLINE bool non_delta() const noexcept {
        return (ray_tag & 0x40000000) == 0;      // check bit 30
    }

    CPT_CPU_GPU_INLINE void set_delta(bool v) noexcept {
        ray_tag &= 0xbfffffff;      // clear bit 30
        ray_tag |= uint32_t(v) << 30;
    }

    CPT_CPU_GPU_INLINE void reset() {
        clr_hit();
        set_hit_index(0);
    }
};
