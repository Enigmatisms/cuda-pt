/**
 * Simplified primitive ray intersection implementation
 * without GPU variant
 * @author: Qianyue He
 * @date:   2024.10.5
*/

#pragma once
#include <limits>
#include "core/aos.cuh"
#include "core/so3.cuh"
#include "core/ray.cuh"
#include "core/interaction.cuh"

using SharedVec3Ptr = Vec3 (*)[32];
using SharedVec2Ptr = Vec2 (*)[32];
using ConstSharedVec3Ptr = const Vec3 (*)[32];
using ConstSharedVec2Ptr = const Vec2 (*)[32];

// #define TRIANGLE_ONLY

// All static
class Primitive {
private:
    CPT_GPU_INLINE static float intersect_sphere(
        const Ray& ray,
        const PrecomputedArray& verts, 
        int index,
        float& solved_u,
        float& solved_v,
        float min_range = EPSILON, 
        float max_range = std::numeric_limits<float>::infinity()
    ) {
        Vec4 center_r = verts.x(index);
        auto op = Vec3(center_r.x(), center_r.y(), center_r.z()) - ray.o; 
        float b = op.dot(ray.d);
        float determinant = b * b - op.dot(op) + center_r.w() * center_r.w(), result = 0;
        if (determinant >= 0) {
            determinant = sqrtf(determinant);
            result = (b - determinant > min_range) ? b - determinant : 0;
            result = (result == 0 && b + determinant > min_range) ? b + determinant : result;
        }
        // currently, sphere does not support UV coords
        solved_u = 0;
        solved_v = 0;
        return result;
    }

    CPT_GPU_INLINE static float intersect_triangle(
        const Ray& ray,
        const PrecomputedArray& verts, 
        int index,
        float& solved_u,
        float& solved_v,
        float min_range = EPSILON, 
        float max_range = std::numeric_limits<float>::infinity()
    ) {
        // solve a linear equation, the current solution is inlined
        auto v1 = verts.y(index), v2 = verts.z(index), anchor = verts.x(index);
        // use precomputed 
        Vec3 v = ray.o - Vec3(anchor.x(), anchor.y(), anchor.z()), 
        temp1(
            fmaf(v2.y(), -ray.d.z(), ray.d.y() * v2.z()),
            fmaf(-ray.d.x(), v2.z(), v2.x() * ray.d.z()),
            fmaf(v2.x(), -ray.d.y(), ray.d.x() * v2.y())
        ),
        temp2(
            fmaf(-ray.d.y(), v1.z(), v1.y() * ray.d.z()),
            fmaf(v1.x(), -ray.d.z(), ray.d.x() * v1.z()),
            fmaf(-ray.d.x(), v1.y(), v1.x() * ray.d.y())
        );

        float inv_det = 1.f / (temp1.x() * v1.x() + temp2.x() * v2.x() - anchor.w() * ray.d.x());
        Vec3 solution(temp1.dot(v) * inv_det, temp2.dot(v) * inv_det, (anchor.w() * v.x() + v1.w() * v.y() + v2.w() * v.z()) * inv_det);

        bool valid    = (solution.x() > 0 && solution.y() > 0 && solution.x() + solution.y() < 1 &&
                            solution.z() > EPSILON && solution.z() < max_range);
        solved_u = solution.x();
        solved_v = solution.y();
        return solution.z() * valid;
    }
public:
    CPT_GPU static float intersect(
        const Ray& ray,
        const PrecomputedArray& verts, 
        int index,
        float& solved_u,
        float& solved_v,
        bool is_mesh = true,
        float min_range = EPSILON, 
        float max_range = std::numeric_limits<float>::infinity()
    ) {
#ifdef TRIANGLE_ONLY
        return intersect_triangle(ray, verts, index, solved_u, solved_v, min_range, max_range);
#else
        if (is_mesh) {
            return intersect_triangle(ray, verts, index, solved_u, solved_v, min_range, max_range);
        } else {
            return intersect_sphere(ray, verts, index, solved_u, solved_v, min_range, max_range);
        }
#endif
    }

    CPT_GPU_INLINE static Interaction get_interaction(
        const PrecomputedArray& verts, 
        const ArrayType<Vec3>& norms, 
        const ArrayType<Vec2>& uvs, 
        Vec3&& hit_pos,
        float u,
        float v,
        int index,
        bool is_mesh = true
    ) {
#ifdef TRIANGLE_ONLY
        float diff_x = 1.f - u, diff_y = 1.f - v;
        return Interaction((
            norms.x(index) * diff_x * diff_y + \
            norms.y(index) * u * diff_y + \
            norms.z(index) * v * diff_x).normalized(),
            uvs.x(index) * diff_x * diff_y + \
            uvs.y(index) * u * diff_y + \
            uvs.z(index) * v * diff_x
        );
#else
        if (is_mesh) {
            float diff_x = 1.f - u, diff_y = 1.f - v;
            return Interaction((
                norms.x(index) * diff_x * diff_y + \
                norms.y(index) * u * diff_y + \
                norms.z(index) * v * diff_x).normalized(),
                uvs.x(index) * diff_x * diff_y + \
                uvs.y(index) * u * diff_y + \
                uvs.z(index) * v * diff_x
            );
        } else {
            return Interaction(
                (hit_pos - verts.x_clipped(index)).normalized(), Vec2(0, 0)
            );
        }
#endif
    }
};