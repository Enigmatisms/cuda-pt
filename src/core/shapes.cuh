/**
 * The definition of AABB, used both in BVH 
 * and plain-old ray intersection
*/

#pragma once
#include <limits>
#include <variant>
#include "core/aos.cuh"
#include "core/aabb.cuh"

class SphereShape {
public:
    int obj_id;
    CPT_CPU SphereShape(int _obj_id = -1): obj_id(_obj_id) {}
};

class TriangleShape {
public:
    int obj_id;
    CPT_CPU TriangleShape(int _obj_id = -1): obj_id(_obj_id) {}
};

/**
 * Helper class for simpler Shape -> AABB construction
*/
class ShapeAABBVisitor {
private:
    const PrecomputedArray& verts;
    mutable AABB* aabb_ptr;
    int index;
public:
    CPT_CPU ShapeAABBVisitor(
        const PrecomputedArray& verts,
        AABB* aabb
    ): verts(verts), aabb_ptr(aabb), index(0) {}

    CPT_CPU void operator()(const TriangleShape& shape) const { 
        auto anchor = verts.x_clipped(index);
        aabb_ptr[index] = AABB(anchor, anchor + verts.y_clipped(index), anchor + verts.z_clipped(index), shape.obj_id, -1);
    }

    CPT_CPU void operator()(const SphereShape& shape) const { 
        auto center_r = verts.x(index);
        Vec3 center   = Vec3(center_r.x(), center_r.y(), center_r.z());
        aabb_ptr[index] = AABB(center - center_r.w(), center + center_r.w(), -shape.obj_id - 1, -1);
    }

    CPT_CPU void set_index(int i)        noexcept { this->index = i; }
};

using Shape = std::variant<TriangleShape, SphereShape>;