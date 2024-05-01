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
    int vt_id;
    int nm_id;
    int uv_id;
    int ob_id;

    CPT_CPU_GPU SphereShape(): vt_id(-1), nm_id(-1), uv_id(-1), ob_id(-1) {}

    CPT_CPU_GPU SphereShape(int _vt_id, int _nm_id, int _uv_id, int _ob_id): 
        vt_id(_vt_id), nm_id(_nm_id), uv_id(_uv_id), ob_id(_ob_id) {}

    // For sphere, uv coordinates is not supported
    CPT_CPU_GPU float intersect(
        const Ray& ray,
        ConstPrimPtr verts, 
        ConstPrimPtr materials, 
        ConstPrimPtr /* normal (useless) */, 
        ConstUVPtr /* uv (useless) */,
        Interaction& it,
        float min_range = epsilon, float max_range = std::numeric_limits<float>::infinity()
    ) const {
        auto op = verts->x[vt_id] - ray.o; 
        float b = op.dot(ray.d);
        float determinant = b * b - op.dot(op) + verts->y[vt_id].x(), result = 0;
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
    int vt_id;
    int nm_id;
    int uv_id;
    int ob_id;
    CPT_CPU_GPU TriangleShape(): vt_id(-1), nm_id(-1), uv_id(-1), ob_id(-1) {}

    CPT_CPU_GPU TriangleShape(int _vt_id, int _nm_id, int _uv_id, int _ob_id): 
        vt_id(_vt_id), nm_id(_nm_id), uv_id(_uv_id), ob_id(_ob_id) {}

    CPT_CPU_GPU float intersect(
        const Ray& ray,
        ConstPrimPtr verts, 
        ConstPrimPtr materials, 
        ConstPrimPtr norms, 
        ConstUVPtr uvs,
        Interaction& it,
        float min_range = epsilon, float max_range = std::numeric_limits<float>::infinity()
    ) const {
        // solve a linear equation
        auto anchor = verts->x[vt_id], v1 = verts->y[vt_id] - anchor, v2 = verts->z[vt_id] - anchor;
        SO3 M(v1, v2, -ray.d, false);       // column wise input
        auto solution = M.inverse().rotate(ray.o - anchor);
        float diff_x = 1.f - solution.x(), diff_y = 1.f - solution.y();
        auto lerp_normal = norms->x[nm_id] * diff_x * diff_y + \
                           norms->y[nm_id] * solution.x() * diff_y + \
                           norms->z[nm_id] * solution.y() * diff_x;
        lerp_normal.normalize();
        auto uv = uvs->x[nm_id] * diff_x * diff_y + \
                  uvs->y[nm_id] * solution.x() * diff_y + \
                  uvs->z[nm_id] * solution.y() * diff_x;

        it.shading_norm = lerp_normal;
        it.uv_coord     = uv;
        it.valid        = (solution.x() > 0 && solution.y() > 0 && solution.x() + solution.y() < 1 &&
                            solution.z() > epsilon && solution.z() < max_range);
        return it.valid - 1 + solution.z() * it.valid;
    }
};