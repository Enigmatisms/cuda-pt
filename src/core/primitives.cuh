/**
 * TODO:
 * 我说一下这里的实现：
 *      1. 去 variant 化，variant 搞成一个分支，以便我们随时可以切回来对比
 *      2. 用 AABB（对于无BVH intersect）或者 BVH 的 AABB 存带符号的obj_id，正为三角形，负为球体
 *          intersect 函数需要输入是否是球体的指示标
 *      3. intersect full 进行简化，不再重算。object_id 的获取也简化，不再重取，直接通过 AABB 获取
 *          prim2obj 应该删除，AABB 完全能存下
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
    CPT_CPU_GPU_INLINE static float intersect_sphere(
        const Ray& ray,
        const ArrayType<Vec3>& verts, 
        int index,
        float& solved_u,
        float& solved_v,
        float min_range = EPSILON, 
        float max_range = std::numeric_limits<float>::infinity()
    ) {
        auto op = verts.x(index) - ray.o; 
        auto _pt = verts.x(index);
        float b = op.dot(ray.d);
        float determinant = b * b - op.dot(op) + verts.y(index).x() * verts.y(index).x(), result = 0;
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

    CPT_CPU_GPU_INLINE static float intersect_triangle(
        const Ray& ray,
        const ArrayType<Vec3>& verts, 
        int index,
        float& solved_u,
        float& solved_v,
        float min_range = EPSILON, 
        float max_range = std::numeric_limits<float>::infinity()
    ) {
        // solve a linear equation
        // TODO: anchor is useless, we can therefore only store two float3 (diff) for a triangle
        // padding will be natural
        auto anchor = verts.x(index), v1 = verts.y(index) - anchor, v2 = verts.z(index) - anchor;
        SO3 M(v1, v2, -ray.d, false);       // column wise input
        auto solution = M.inverse_transform(ray.o - anchor);
        bool valid    = (solution.x() > 0 && solution.y() > 0 && solution.x() + solution.y() < 1 &&
                            solution.z() > EPSILON && solution.z() < max_range);
        solved_u = solution.x();
        solved_v = solution.y();
        return solution.z() * valid;
    }
public:
    CPT_CPU_GPU static float intersect(
        const Ray& ray,
        const ArrayType<Vec3>& verts, 
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

    CPT_CPU_GPU_INLINE static Interaction get_interaction(
        const ArrayType<Vec3>& verts, 
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
                (verts.x(index) - hit_pos).normalized(), Vec2(0, 0)
            );
        }
#endif
    }
};