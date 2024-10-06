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
    const ArrayType<Vec3>& verts;
    mutable AABB* aabb_ptr;
    int index;
public:
    CPT_CPU ShapeAABBVisitor(
        const ArrayType<Vec3>& verts,
        AABB* aabb
    ): verts(verts), aabb_ptr(aabb), index(0) {}

    CPT_CPU void operator()(const TriangleShape& shape) const { 
        aabb_ptr[index] = AABB(verts.x(index), verts.y(index), verts.z(index), shape.obj_id, -1);
    }

    CPT_CPU void operator()(const SphereShape& shape) const { 
        aabb_ptr[index] = AABB(verts.x(index) - verts.y(index).x(), verts.x(index) + verts.y(index).x(), -shape.obj_id - 1, -1);
    }

    CPT_CPU void set_index(int i)        noexcept { this->index = i; }
};

using Shape = std::variant<TriangleShape, SphereShape>;