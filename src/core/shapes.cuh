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
#include <variant/variant.h>
#include "core/aos.cuh"
#include "core/so3.cuh"
#include "core/ray.cuh"
#include "core/interaction.cuh"

using SharedVec3Ptr = Vec3 (*)[32];
using SharedVec2Ptr = Vec2 (*)[32];
using ConstSharedVec3Ptr = const Vec3 (*)[32];
using ConstSharedVec2Ptr = const Vec2 (*)[32];

class AABB {
public:
    Vec3 mini;
    CUDA_PT_PADDING(1, 1)
    Vec3 maxi;
    CUDA_PT_PADDING(1, 2)
public:
    CPT_CPU_GPU AABB(): mini(), maxi() {}

    template <typename V1Type, typename V2Type>
    CPT_CPU_GPU AABB(V1Type&& _mini, V2Type&& _maxi):
        mini(std::forward<V1Type>(_mini)), maxi(std::forward<V2Type>(_maxi)) {}

    CPT_CPU_GPU AABB(const Vec3& p1, const Vec3& p2, const Vec3& p3) {
        mini = p1.minimize(p2).minimize(p3);
        mini -= AABB_EPS;
        maxi = p1.maximize(p2).maximize(p3);
        maxi += AABB_EPS;
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

    CONDITION_TEMPLATE(AABBType, AABB)
    CPT_CPU_GPU AABB& operator += (AABBType&& _aabb) noexcept {
        mini = mini.minimize(_aabb.mini);
        maxi = maxi.maximize(_aabb.maxi);
        return *this;
    }

    CONDITION_TEMPLATE(AABBType, AABB)
    CPT_CPU_GPU AABB operator+ (AABBType&& _aabb) const noexcept {
        return AABB(
            mini.minimize(_aabb.mini),
            maxi.maximize(_aabb.maxi)
        );
    }

    CPT_GPU_INLINE void copy_from(const AABB& other) {
        FLOAT4(mini) = CONST_FLOAT4(other.mini);
        FLOAT4(maxi) = CONST_FLOAT4(other.maxi); // Load last two elements of second Vec3
    }

    CPT_CPU_GPU_INLINE float area() const {
        Vec3 diff = maxi - mini;
        return 2.f * (diff.x() * diff.y() + diff.y() * diff.z() + diff.x() * diff.z());
    }

    CPT_CPU_GPU_INLINE void clear() {
        mini.fill(1e4);
        maxi.fill(-1e4);
    }
};

struct AABBWrapper {
    AABB aabb;
    float4 _padding;            // padding is here to avoid bank conflict
};

using ConstAABBPtr = const AABB* const;
using ConstAABBWPtr = const AABBWrapper* const;

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
