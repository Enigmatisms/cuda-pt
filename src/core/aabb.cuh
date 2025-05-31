// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * @author Qianyue He
 * @brief Axis Aligned Bounding Box
 * @date 2025.01.06
 */

#pragma once
#include "core/ray.cuh"

class AABB {
  public:
    Vec3 mini;
    CUDA_PT_SINGLE_PADDING(1) // used as prim_idx, for BVH
    Vec3 maxi;
    CUDA_PT_SINGLE_PADDING(2) // used as obj_idx for BVH
  public:
    CPT_CPU_GPU AABB() {}
    CPT_CPU_GPU AABB(int p1, int p2)
        : mini(), __bytes1(p1), maxi(), __bytes2(p2) {}
    CPT_CPU_GPU AABB(float min_v, float max_v, int p1, int p2)
        : mini(min_v), __bytes1(p1), maxi(max_v), __bytes2(p2) {}

    CONDITION_TEMPLATE_2(V1Type, V2Type, Vec3)
    CPT_CPU_GPU AABB(V1Type &&_mini, V2Type &&_maxi, int _obj_idx,
                     int _prim_idx)
        : mini(std::forward<V1Type>(_mini)), __bytes1(_obj_idx),
          maxi(std::forward<V2Type>(_maxi)), __bytes2(_prim_idx) {}

    CPT_CPU_GPU AABB(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3,
                     int _obj_idx, int _prim_idx)
        : __bytes1(_obj_idx), __bytes2(_prim_idx) {
        mini = p1.minimize(p2).minimize(p3);
        mini -= AABB_EPS;
        maxi = p1.maximize(p2).maximize(p3);
        maxi += AABB_EPS;
    }

    CPT_CPU Vec3 centroid() const noexcept { return (maxi + mini) * 0.5f; }
    CPT_CPU Vec3 range() const noexcept { return maxi - mini; }

    CPT_GPU bool intersect(Vec3 inv_d, Vec3 o_div, float &t_near) const {
        auto t1s = mini.fmsub(inv_d, o_div);
        inv_d = maxi.fmsub(inv_d, o_div);

        float tmax = 0;
        t1s.min_max(inv_d, t_near, tmax);
        return (tmax > t_near) && (tmax > 0); // local memory access problem
    }

    CPT_GPU bool intersect(const Ray &ray, float &t_near) const {
        auto t2s = ray.d.rcp(), o_div = ray.o * t2s;
        auto t1s = mini.fmsub(t2s, o_div);
        t2s = maxi.fmsub(t2s, o_div);

        float tmax = 0;
        t1s.min_max(t2s, t_near, tmax);
        return (tmax > t_near) && (tmax > 0); // local memory access problem
    }

    CPT_GPU bool intersect(const Ray &ray, float &t_near, float &t_far) const {
        auto t2s = ray.d.rcp(), o_div = ray.o * t2s;
        auto t1s = mini.fmsub(t2s, o_div);
        t2s = maxi.fmsub(t2s, o_div);

        t1s.min_max(t2s, t_near, t_far);
        return (t_far > t_near) && (t_far > 0); // local memory access problem
    }

    CONDITION_TEMPLATE(AABBType, AABB)
    CPT_CPU AABB &operator+=(AABBType &&_aabb) noexcept {
        mini = mini.minimize(_aabb.mini);
        maxi = maxi.maximize(_aabb.maxi);
        return *this;
    }

    CONDITION_TEMPLATE(PointType, Vec3)
    CPT_CPU void extend(PointType &&pt) noexcept {
        mini.minimized(pt);
        maxi.maximized(std::forward<PointType>(pt));
    }

    CONDITION_TEMPLATE(VecType, Vec3)
    CPT_CPU bool covers(VecType &&pt) const noexcept {
#define IN_RANGE(x, x_min, x_max) (x > x_min && x < x_max)
        return IN_RANGE(pt.x(), mini.x(), maxi.x()) &&
               IN_RANGE(pt.y(), mini.y(), maxi.y()) &&
               IN_RANGE(pt.z(), mini.z(), maxi.z());
#undef IN_RANGE
    }

    CPT_CPU void fix_degenerate() {
        constexpr float THRESHOLD = 1e-4f;
#pragma unroll
        for (int i = 0; i < 3; i++) {
            if (maxi[i] - mini[i] < THRESHOLD) {
                maxi[i] += THRESHOLD * 0.5f;
                mini[i] -= THRESHOLD * 0.5f;
            }
        }
    }

    // intersection of two AABB
    CONDITION_TEMPLATE(AABBType, AABB)
    CPT_CPU float intersection_area(AABBType &&_aabb) const noexcept {
        if (range().max_elem() < 0 || _aabb.range().max_elem() < 0)
            return 0;
        auto temp_min = mini.maximize(_aabb.mini);
        auto temp_max = maxi.minimize(_aabb.maxi);
        if (temp_max.x() <= temp_min.x() || temp_max.y() <= temp_min.y() ||
            temp_max.z() <= temp_min.z()) {
            return 0;
        }
        auto diff = temp_max - temp_min;
        return 2.f * (diff.x() * diff.y() + diff.y() * diff.z() +
                      diff.z() * diff.x());
    }

    CPT_GPU_INLINE void copy_from(const AABB &other) {
        FLOAT4(mini) = CONST_FLOAT4(other.mini);
        FLOAT4(maxi) =
            CONST_FLOAT4(other.maxi); // Load last two elements of second Vec3
    }

    // A safe call for area
    CPT_CPU_INLINE float area() const {
        Vec3 diff = maxi - mini;
        if (fabsf(diff.x()) < AABB_INVALID_DIST)
            return 2.f * (diff.x() * diff.y() + diff.y() * diff.z() +
                          diff.x() * diff.z());
        return 0;
    }

    CPT_CPU_INLINE void clear() {
        mini.fill(AABB_INVALID_DIST);
        maxi.fill(-AABB_INVALID_DIST);
    }

    CPT_CPU_GPU_INLINE int obj_idx() const { return __bytes1; }
    CPT_CPU_GPU_INLINE int &obj_idx() { return __bytes1; }

    CPT_CPU_GPU_INLINE int prim_idx() const { return __bytes2; }
    CPT_CPU_GPU_INLINE int &prim_idx() { return __bytes2; }

    CPT_CPU_GPU_INLINE int base() const { return __bytes1; }
    CPT_CPU_GPU_INLINE int &base() { return __bytes1; }

    CPT_CPU_GPU_INLINE int prim_cnt() const { return __bytes2; }
    CPT_CPU_GPU_INLINE int &prim_cnt() { return __bytes2; }

    CONDITION_TEMPLATE_SEP_2(V1Type, V2Type, Vec3, Vec3)
    CPT_CPU bool clip_line_segment(V1Type &&p0, V2Type &&p1, Vec3 &out0,
                                   Vec3 &out1) const {
        // early check: if both endpoints are inside the AABB, no clipping
        // needed.
        if (covers(p0) && covers(p1)) {
            out0 = p0;
            out1 = p1;
            return true;
        }

        // direction vector and initialization of parametric parameters t0
        // (enter) and t1 (exit).
        Vec3 dir = p1 - p0;
        float t0 = 0.0f; // start of line segment (p0)
        float t1 = 1.0f; // end of line segment (p1)

        // process each dimension (x, y, z)
        for (int i = 0; i < 3; ++i) {
            if (dir[i] == 0.0f) {
                // line segment is parallel to this dimension's planes.
                // if the current point is outside, the entire segment is
                // outside.
                if (p0[i] < mini[i] || p0[i] > maxi[i]) {
                    return false;
                }
            } else {
                // calculate parametric values t_min and t_max for intersections
                // with min and max planes, basic ray-AABB intersection
                float inv_d = 1.0f / dir[i];
                float t_min = (mini[i] - p0[i]) * inv_d;
                float t_max = (maxi[i] - p0[i]) * inv_d;

                // swap t_min and t_max if direction is negative (entering from
                // opposite side).
                if (inv_d < 0.0f) {
                    std::swap(t_min, t_max);
                }

                // update global t0 (enter) and t1 (exit) based on current
                // dimension.
                t0 = std::max(t0, t_min);
                t1 = std::min(t1, t_max);

                // early exit if the segment is completely outside the AABB.
                if (t0 > t1) {
                    return false;
                }
            }
        }

        out0 = p0 + dir * t0;
        out1 = p0 + dir * t1;
        return true;
    }
};

struct AABBWrapper {
    AABB aabb;
    float4 _padding; // padding is here to avoid bank conflict
};

using ConstAABBPtr = const AABB *const __restrict__;
using ConstAABBWPtr = const AABBWrapper *const __restrict__;
