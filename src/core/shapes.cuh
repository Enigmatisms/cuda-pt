/**
 * The definition of two common shapes: Sphere & Triangle
 * the classes defined here will contain the primitive ID and object ID
 * primitive ID points to the primitive data, including:
 * (1) vertex id: 3 vec3 (indicating the position of the vertices). For spheres
 * only the first vec3 and the first value of the second vec3 will be used
 * AABB id is the same as vertex id
 * (2) normal id: 3 vec3 for vertex normal, sphere idx will have this idx set to -1
 * (3) uv id: 3 vec2 for UV coordinates,  sphere idx will have this idx set to -1
 * (4) object id: to query material property or texture
 * 
 * method: intersect
*/

#pragma once
#include <limits>
#include "core/soa.cuh"
#include "core/so3.cuh"
#include "core/ray.cuh"
#include "core/interaction.cuh"

static constexpr float epsilon = 1e-4;

using SharedVec3Ptr = Vec3 (*)[32];
using SharedVec2Ptr = Vec2 (*)[32];

class AABB {
public:
    Vec3 mini;
    Vec3 maxi;
public:
    CPT_CPU_GPU AABB(): mini(), maxi() {}

    template <typename V1Type, typename V2Type>
    CPT_CPU_GPU AABB(V1Type&& _mini, V2Type&& _maxi):
        mini(std::forward<V1Type>(_mini)), maxi(std::forward<V2Type>(_maxi)) {}

    CPT_CPU_GPU AABB(const Vec3& p1, const Vec3& p2, const Vec3& p3) {
        mini = p1.minimize(p2).minimize(p3);
        mini -= epsilon;
        maxi = p1.maximize(p2).maximize(p3);
        maxi += epsilon;
    }

    CPT_CPU_GPU Vec3 centroid() const noexcept {return (maxi + mini) * 0.5f;}
    CPT_CPU_GPU Vec3 range()    const noexcept {return maxi - mini;}

    CPT_CPU_GPU bool intersect(const Ray& ray, float& t_near) const {
        auto invDir = 1.0f / ray.d;

        auto t1s = (mini - ray.o) * invDir;
        auto t2s = (maxi - ray.o) * invDir;

        float tmin = t1s.minimize(t2s).max_elem();
        float tmax = t1s.maximize(t2s).min_elem();
        t_near = tmin;
        return tmax > tmin && tmax > 0;
    }
};
using ConstAABBPtr = const AABB* const;

class SphereShape {
public:
    int obj_idx;            // object of the current shape
    CPT_CPU_GPU SphereShape(): obj_idx(-1) {}

    CPT_CPU_GPU SphereShape(int _ob_id): obj_idx(_ob_id) {}

    // For sphere, uv coordinates is not supported
    CPT_CPU_GPU float intersect(
        const Ray& ray,
        SharedVec3Ptr verts, 
        SharedVec3Ptr /* normal (useless) */, 
        SharedVec2Ptr /* uv (useless) */,
        Interaction& it,
        int index,
        float min_range = epsilon, float max_range = std::numeric_limits<float>::infinity()
    ) const {
        auto op = verts[0][index] - ray.o; 
        float b = op.dot(ray.d);
        float determinant = b * b - op.dot(op) + verts[1][index].x() * verts[1][index].x(), result = 0;
        if (determinant >= 0) {
            determinant = sqrtf(determinant);
            result = b - determinant > min_range ? b - determinant : 0;
            result = (result == 0 && b + determinant > min_range) ? b + determinant : result;
        }
        return result;
    }
};

class TriangleShape {
    
public:
    int obj_idx;            // object of the current shape
    CPT_CPU_GPU TriangleShape(): obj_idx(-1) {}

    CPT_CPU_GPU TriangleShape(int _ob_id): obj_idx(_ob_id) {}

    CPT_CPU_GPU float intersect(
        const Ray& ray,
        SharedVec3Ptr verts, 
        SharedVec3Ptr norms, 
        SharedVec2Ptr uvs,
        Interaction& it,
        int index,
        float min_range = epsilon, float max_range = std::numeric_limits<float>::infinity()
    ) const {
        // solve a linear equation
        auto anchor = verts[0][index], v1 = verts[1][index] - anchor, v2 = verts[2][index] - anchor;
        SO3 M(v1, v2, -ray.d, false);       // column wise input
        auto solution = M.inverse().rotate(ray.o - anchor);
        float diff_x = 1.f - solution.x(), diff_y = 1.f - solution.y();
        auto lerp_normal = norms[0][index] * diff_x * diff_y + \
                           norms[1][index] * solution.x() * diff_y + \
                           norms[2][index] * solution.y() * diff_x;
        lerp_normal.normalize();
        auto uv = uvs[0][index] * diff_x * diff_y + \
                  uvs[1][index] * solution.x() * diff_y + \
                  uvs[2][index] * solution.y() * diff_x;

        it.shading_norm = lerp_normal;
        it.uv_coord     = uv;
        it.valid        = (solution.x() > 0 && solution.y() > 0 && solution.x() + solution.y() < 1 &&
                            solution.z() > epsilon && solution.z() < max_range);
        return solution.z() * it.valid;
    }
};

class ShapeVisitor {
private:
    const Ray* ray;
    SharedVec3Ptr verts; 
    SharedVec3Ptr norms; 
    SharedVec2Ptr uvs;
    mutable Interaction* it;        // apply_visitor seems to only work for const member function
    int index;
    float min_range;
    float max_range;
public:
    CPT_CPU_GPU ShapeVisitor(
        SharedVec3Ptr _verts, 
        SharedVec3Ptr _norms, 
        SharedVec2Ptr _uvs,
        const Ray* _ray,
        Interaction* _it,
        int _index,
        float _min_range = epsilon, float _max_range = std::numeric_limits<float>::infinity()
    ): ray(_ray), verts(_verts), norms(_norms), uvs(_uvs), it(_it), 
       index(_index), min_range(_min_range), max_range(_max_range) {}

    template <typename ShapeType>
    CPT_CPU_GPU_INLINE float operator()(const ShapeType& shape) const { 
        return shape.intersect(*ray, verts, norms, uvs, *it, index, min_range, max_range); 
    }

    CPT_CPU_GPU_INLINE void set_index(int i)        noexcept { this->index = i; }
    CPT_CPU_GPU_INLINE void set_it(Interaction* it) noexcept { this->it    = it; }

    CPT_CPU_GPU_INLINE void set_min_range(int min_r) noexcept { this->min_range = min_r; }
    CPT_CPU_GPU_INLINE void set_max_range(int max_r) noexcept { this->max_range = max_r; }
};
