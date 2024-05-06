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

class AABB {
public:
    Vec3 mini;
    Vec3 maxi;
public:
    CPT_CPU_GPU AABB(): mini(), maxi() {}

    template <typename V1Type, typename V2Type>
    CPT_CPU_GPU AABB(V1Type&& _mini, V2Type&& _maxi):
        mini(std::forward<V1Type>(_mini)), maxi(std::forward<V2Type>(_maxi)) {}

    CPT_CPU_GPU Vec3 centroid() const noexcept {return (maxi + mini) * 0.5f;}
    CPT_CPU_GPU Vec3 range()    const noexcept {return maxi - mini;}

    CPT_CPU_GPU bool intersect(const Ray& ray) const {
        float tmin = -std::numeric_limits<float>::infinity();
        float tmax = std::numeric_limits<float>::infinity();
        auto invDir = 1.0f / ray.d;

        auto t1s = (mini - ray.o) * invDir;
        auto t2s = (maxi - ray.o) * invDir;

        tmin = max(tmin, t1s.minimize(t2s));
        tmax = min(tmax, t1s.maximize(t2s));

        return tmax >= tmin && tmax >= 0;
    }
};

class SphereShape {
public:
    bool    has_norm;           // whether triangle vertices has normal 
    bool    has_uv;             // whether triangle vertices has uv coords
    int16_t obj_idx;            // object of the current shape
    CPT_CPU_GPU SphereShape(): has_norm(false), has_uv(false), obj_idx(-1) {}

    CPT_CPU_GPU SphereShape(int _ob_id, bool _hn = false, bool _huv = false): 
        has_norm(_hn), has_uv(_huv), obj_idx(_ob_id) {}

    // For sphere, uv coordinates is not supported
    CPT_CPU_GPU float intersect(
        const Ray& ray,
        ConstPrimPtr verts, 
        ConstPrimPtr /* normal (useless) */, 
        ConstUVPtr /* uv (useless) */,
        Interaction& it,
        int index,
        float min_range = epsilon, float max_range = std::numeric_limits<float>::infinity()
    ) const {
        auto op = verts->x[index] - ray.o; 
        float b = op.dot(ray.d);
        float determinant = b * b - op.dot(op) + verts->y[index].x(), result = 0;
        if (determinant >= 0) {
            determinant = sqrtf(determinant);
            float result = b - determinant > min_range ? b - determinant : 0;
            result = (result > 0 && b + determinant > min_range) ? b + determinant : result;
        }
        return result;
    }
};

class TriangleShape {
    
public:
    bool    has_norm;           // whether triangle vertices has normal 
    bool    has_uv;             // whether triangle vertices has uv coords
    int16_t obj_idx;            // object of the current shape
    CPT_CPU_GPU TriangleShape(): has_norm(false), has_uv(false), obj_idx(-1) {}

    CPT_CPU_GPU TriangleShape(int _ob_id, bool _hn = false, bool _huv = false): 
        has_norm(_hn), has_uv(_huv), obj_idx(_ob_id) {}

    CPT_CPU_GPU float intersect(
        const Ray& ray,
        ConstPrimPtr verts, 
        ConstPrimPtr norms, 
        ConstUVPtr uvs,
        Interaction& it,
        int index,
        float min_range = epsilon, float max_range = std::numeric_limits<float>::infinity()
    ) const {
        // solve a linear equation
        auto anchor = verts->x[index], v1 = verts->y[index] - anchor, v2 = verts->z[index] - anchor;
        SO3 M(v1, v2, -ray.d, false);       // column wise input
        auto solution = M.inverse().rotate(ray.o - anchor);
        float diff_x = 1.f - solution.x(), diff_y = 1.f - solution.y();
        auto lerp_normal = norms->x[index] * diff_x * diff_y + \
                           norms->y[index] * solution.x() * diff_y + \
                           norms->z[index] * solution.y() * diff_x;
        lerp_normal.normalize();
        auto uv = uvs->x[index] * diff_x * diff_y + \
                  uvs->y[index] * solution.x() * diff_y + \
                  uvs->z[index] * solution.y() * diff_x;

        it.shading_norm = lerp_normal;
        it.uv_coord     = uv;
        it.valid        = (solution.x() > 0 && solution.y() > 0 && solution.x() + solution.y() < 1 &&
                            solution.z() > epsilon && solution.z() < max_range);
        return it.valid - 1 + solution.z() * it.valid;
    }
};