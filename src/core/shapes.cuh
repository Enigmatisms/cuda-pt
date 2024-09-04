/**
 * The definition of AABB, used both in BVH 
 * and plain-old ray intersection
*/

#pragma once
#include <limits>
#include <variant/variant.h>
#include "core/aos.cuh"
#include "core/so3.cuh"
#include "core/ray.cuh"
#include "core/aabb.cuh"
#include "core/interaction.cuh"

using SharedVec3Ptr = Vec3 (*)[32];
using SharedVec2Ptr = Vec2 (*)[32];
using ConstSharedVec3Ptr = const Vec3 (*)[32];
using ConstSharedVec2Ptr = const Vec2 (*)[32];


class SphereShape {
public:
    // TODO: this obj_idx might be deprecated in the future
    int obj_idx;            // object of the current shape
    CPT_CPU_GPU SphereShape(): obj_idx(-1) {}

    CPT_CPU_GPU SphereShape(int _ob_id): obj_idx(_ob_id) {}

    // For sphere, uv coordinates is not supported
    CPT_CPU_GPU float intersect(
        const Ray& ray,
        const ArrayType<Vec3>& verts, 
        int index,
        float min_range = EPSILON, float max_range = std::numeric_limits<float>::infinity()
    ) const {
        auto op = verts.x(index) - ray.o; 
        float b = op.dot(ray.d);
        float determinant = b * b - op.dot(op) + verts.y(index).x() * verts.y(index).x(), result = 0;
        if (determinant >= 0) {
            determinant = sqrtf(determinant);
            result = b - determinant > min_range ? b - determinant : 0;
            result = (result == 0 && b + determinant > min_range) ? b + determinant : result;
        }
        return result;
    }

    CPT_CPU_GPU Interaction intersect_full(
        const Ray& ray,
        const ArrayType<Vec3>& verts, 
        const ArrayType<Vec3>& norms, 
        const ArrayType<Vec2>& uvs, 
        int index
    ) const {
        auto op = verts.x(index) - ray.o; 
        float b = op.dot(ray.d);
        float determinant = sqrtf(b * b - op.dot(op) + verts.y(index).x() * verts.y(index).x());
        float result = b - determinant > EPSILON ? b - determinant : 0;
        result = (result == 0 && b + determinant > EPSILON) ? b + determinant : result;
        return Interaction((ray.d * result - op).normalized(), Vec2(0, 0));
    }
};

class TriangleShape {
    
public:
    // TODO: this obj_idx might be deprecated in the future
    int obj_idx;            // object of the current shape
    CPT_CPU_GPU TriangleShape(): obj_idx(-1) {}

    CPT_CPU_GPU TriangleShape(int _ob_id): obj_idx(_ob_id) {}

    CPT_CPU_GPU float intersect(
        const Ray& ray,
        const ArrayType<Vec3>& verts, 
        int index,
        float min_range = EPSILON, float max_range = std::numeric_limits<float>::infinity()
    ) const {
        // solve a linear equation
        auto anchor = verts.x(index), v1 = verts.y(index) - anchor, v2 = verts.z(index) - anchor;
        SO3 M(v1, v2, -ray.d, false);       // column wise input
        auto solution = M.inverse_transform(ray.o - anchor);
        bool valid    = (solution.x() > 0 && solution.y() > 0 && solution.x() + solution.y() < 1 &&
                            solution.z() > EPSILON && solution.z() < max_range);
        return solution.z() * valid;
    }

    CPT_CPU_GPU Interaction intersect_full(
        const Ray& ray,
        const ArrayType<Vec3>& verts, 
        const ArrayType<Vec3>& norms, 
        const ArrayType<Vec2>& uvs, 
        int index
    ) const {
        // solve a linear equation
        auto anchor = verts.x(index), v1 = verts.y(index) - anchor, v2 = verts.z(index) - anchor;
        SO3 M(v1, v2, -ray.d, false);       // column wise input
        auto solution = M.inverse().rotate(ray.o - anchor);
        float diff_x = 1.f - solution.x(), diff_y = 1.f - solution.y();
        return Interaction((
            norms.x(index) * diff_x * diff_y + \
            norms.y(index) * solution.x() * diff_y + \
            norms.z(index) * solution.y() * diff_x).normalized(),
            uvs.x(index) * diff_x * diff_y + \
            uvs.y(index) * solution.x() * diff_y + \
            uvs.z(index) * solution.y() * diff_x
        );
    }
};

class ShapeIntersectVisitor {
private:
    const Ray& ray;
    const ArrayType<Vec3>& verts; 
    int index;
    float max_range;    
public:
    CPT_CPU_GPU ShapeIntersectVisitor(
        const ArrayType<Vec3>& _verts, 
        const Ray& _ray,
        int _index,
        float _max_range = std::numeric_limits<float>::infinity()
    ): ray(_ray), verts(_verts), 
       index(_index), max_range(_max_range) {}

    template <typename ShapeType>
    CPT_CPU_GPU_INLINE float operator()(const ShapeType& shape) const { 
        return shape.intersect(ray, verts, index, EPSILON, max_range); 
    }

    CPT_CPU_GPU_INLINE void set_index(int i)        noexcept { this->index = i; }
    CPT_CPU_GPU_INLINE void set_max_range(int max_r) noexcept { this->max_range = max_r; }
};

class ShapeExtractVisitor {
private:
    const Ray& ray;
    const ArrayType<Vec3>& verts; 
    const ArrayType<Vec3>& norms; 
    const ArrayType<Vec2>& uvs; 
    int index;
public:
    CPT_CPU_GPU ShapeExtractVisitor(
        const ArrayType<Vec3>& _verts, 
        const ArrayType<Vec3>& _norms, 
        const ArrayType<Vec2>& _uvs, 
        const Ray& _ray,
        int _index
    ): ray(_ray), verts(_verts), norms(_norms), uvs(_uvs), index(_index) {}

    template <typename ShapeType>
    CPT_CPU_GPU_INLINE Interaction operator()(const ShapeType& shape) const { 
        return shape.intersect_full(ray, verts, norms, uvs, index); 
    }

    CPT_CPU_GPU_INLINE void set_index(int i) noexcept { this->index = i; }
};

/**
 * Helper class for simpler Shape -> AABB construction
*/
class ShapeAABBVisitor {
private:
    const ArrayType<Vec3>& verts;
    mutable AABB* aabb_ptr;
    int index;
public:
    CPT_CPU ShapeAABBVisitor(
        const ArrayType<Vec3>& verts,
        AABB* aabb
    ): verts(verts), aabb_ptr(aabb), index(0) {}

    CPT_CPU void operator()(const TriangleShape& shape) const { 
        aabb_ptr[index] = AABB(verts.x(index), verts.y(index), verts.z(index));
    }

    CPT_CPU void operator()(const SphereShape& shape) const { 
        aabb_ptr[index] = AABB(verts.x(index) - verts.y(index).x(), verts.x(index) + verts.y(index).x());
    }

    CPT_CPU void set_index(int i)        noexcept { this->index = i; }
};

using Shape = variant::variant<TriangleShape, SphereShape>;
using ConstShapePtr = const Shape* const;
