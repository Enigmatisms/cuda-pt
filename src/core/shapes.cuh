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
#include "core/ray.cuh"

template <typename Ty = float>
class SphereShape {
public:
    int vt_id;
    int nm_id;
    int uv_id;
    int ob_id;

    CPT_GPU SphereShape(): vt_id(-1), nm_id(-1), uv_id(-1), ob_id(-1) {}

    CPT_GPU SphereShape(int _vt_id, int _nm_id, int _uv_id, int _ob_id): 
        vt_id(_vt_id), nm_id(_nm_id), uv_id(_uv_id), ob_id(_ob_id) {}

    CPT_GPU Ty intersect(
        const Ray& ray,
        ConstPrimPtr<Ty> verts, 
        ConstPrimPtr<Ty> materials, 
        ConstPrimPtr<Ty> /* normal (useless) */, 
        ConstUVPtr<Ty> /* uv (useless) */,
        Ty min_range = 0, Ty max_range = std::numeric_limits<Ty>::infinity()
    ) const {
        auto op = verts.x[vt_id] - r.o; 
        Ty b = op.dot(r.d);
        Ty determinant = b * b - op.dot(op) + verts.y[vt_id].x;
        if (determinant >= 0) {
            determinant = sqrtf(determinant);
            Ty result = b - determinant > min_range ? b - determinant : 0;
            result = (result > 0 && b + determinant > min_range) ? b + determinant : result;
        }
        return 0;
    }
};

template <typename Ty = float>
class TriangleShape {
public:
    int vt_id;
    int nm_id;
    int uv_id;
    int ob_id;
    CPT_GPU TriangleShape(): vt_id(-1), nm_id(-1), uv_id(-1), ob_id(-1) {}

    CPT_GPU TriangleShape(int _vt_id, int _nm_id, int _uv_id, int _ob_id): 
        vt_id(_vt_id), nm_id(_nm_id), uv_id(_uv_id), ob_id(_ob_id) {}

    CPT_GPU Ty intersect(
        const Ray& ray,
        ConstPrimPtr<Ty> verts, 
        ConstPrimPtr<Ty> materials, 
        ConstPrimPtr<Ty> norms, 
        ConstUVPtr<Ty> uvs,
        Ty min_range = 0, Ty max_range = std::numeric_limits<Ty>::infinity()
    ) const {
        // solve a linear equation

    }
};