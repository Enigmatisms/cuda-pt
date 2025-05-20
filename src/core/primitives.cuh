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
 * @author: Qianyue He
 * @brief Simplified primitive ray intersection implementation
 * without GPU variant
 * @date:   2024.10.5
 */

#pragma once
#include "core/aos.cuh"
#include "core/defines.cuh"
#include "core/interaction.cuh"
#include "core/ray.cuh"
#include "core/so3.cuh"
#include <limits>

class Primitive {
  private:
    CPT_GPU_INLINE static float intersect_sphere(
        const Ray &ray, const PrecomputedArray &verts, const int index,
        float &solved_u, float &solved_v, const float min_range = EPSILON,
        const float max_range = std::numeric_limits<float>::infinity()) {
        const Vec4 center_r = verts.x(index);
        const auto op = Vec3(center_r.x(), center_r.y(), center_r.z()) - ray.o;
        const float b = op.dot(ray.d);
        float determinant = b * b - op.dot(op) + center_r.w() * center_r.w(),
              result = 0;
        if (determinant >= 0) {
            determinant = sqrtf(determinant);
            result = (b - determinant > min_range) ? b - determinant : 0;
            result = (result == 0 && b + determinant > min_range)
                         ? b + determinant
                         : result;
        }
        // currently, sphere does not support UV coords
        solved_u = 0;
        solved_v = 0;
        return result;
    }

    CPT_GPU_INLINE static float intersect_triangle(
        const Ray &ray, const PrecomputedArray &verts, const int index,
        float &solved_u, float &solved_v, const float min_range = EPSILON,
        const float max_range = std::numeric_limits<float>::infinity()) {
        // solve a linear equation, the current solution is inlined
        float4 temp;
        Vec3 v, temp1, temp2;
        {
            const float4 v1 = verts.y(index), v2 = verts.z(index);
            temp = verts.x(index);
            v = ray.o - Vec3(temp.x, temp.y, temp.z);
            temp1 = Vec3(fmaf(v2.y, -ray.d.z(), ray.d.y() * v2.z),
                         fmaf(-ray.d.x(), v2.z, v2.x * ray.d.z()),
                         fmaf(v2.x, -ray.d.y(), ray.d.x() * v2.y));
            temp2 = Vec3(fmaf(-ray.d.y(), v1.z, v1.y * ray.d.z()),
                         fmaf(v1.x, -ray.d.z(), ray.d.x() * v1.z),
                         fmaf(-ray.d.x(), v1.y, v1.x * ray.d.y()));
            temp.x = v1.x * temp1.x() + v2.x * temp2.x() - temp.w * ray.d.x(),
            temp.y = v1.w;
            temp.z = v2.w;
        }

        const float inv_det = 1.f / temp.x;
        v = Vec3(temp1.dot(v) * inv_det, temp2.dot(v) * inv_det,
                 (temp.w * v.x() + temp.y * v.y() + temp.z * v.z()) * inv_det);

        solved_u = v.x();
        solved_v = v.y();
        return v.z() * (v.x() > 0 && v.y() > 0 && v.x() + v.y() < 1 &&
                        v.z() > EPSILON && v.z() < max_range);
    }

  public:
    CPT_GPU static float
    intersect(const Ray &ray, const PrecomputedArray &verts, int index,
              float &solved_u, float &solved_v, bool is_mesh = true,
              float min_range = EPSILON,
              float max_range = std::numeric_limits<float>::infinity()) {
#ifdef TRIANGLE_ONLY
        return intersect_triangle(ray, verts, index, solved_u, solved_v,
                                  min_range, max_range);
#else
        if (is_mesh) {
            return intersect_triangle(ray, verts, index, solved_u, solved_v,
                                      min_range, max_range);
        } else {
            return intersect_sphere(ray, verts, index, solved_u, solved_v,
                                    min_range, max_range);
        }
#endif
    }

    CPT_GPU_INLINE static Interaction
    get_interaction(const PrecomputedArray &verts, const NormalArray &norms,
                    const ConstBuffer<PackedHalf2> &uvs, Vec3 &&hit_pos,
                    float u, float v, int index, bool is_mesh = true) {
#ifdef TRIANGLE_ONLY
        return Interaction(norms.eval(index, u, v), uvs[index].lerp(u, v));
#else
        if (is_mesh) {
            return Interaction(norms.eval(index, u, v), uvs[index].lerp(u, v));
        } else {
            return Interaction((hit_pos - verts.x_clipped(index)).normalized(),
                               Vec2Half(0, 0));
        }
#endif
    }
};
