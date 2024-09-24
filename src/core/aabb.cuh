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
#include "core/ray.cuh"

class AABB {
public:
    Vec3 mini;
    CUDA_PT_SINGLE_PADDING(1)           // used as prim_idx, for BVH
    Vec3 maxi;
    CUDA_PT_SINGLE_PADDING(2)           // used as obj_idx for BVH
public:
    CPT_CPU_GPU AABB() {}
    CPT_CPU_GPU AABB(int p1, int p2): mini(), __bytes1(p1), maxi(), __bytes2(p2) {}
    CPT_CPU_GPU AABB(float min_v, float max_v, int p1, int p2): 
        mini(min_v), __bytes1(p1), maxi(max_v), __bytes2(p2) {}

    CONDITION_TEMPLATE_2(V1Type, V2Type, Vec3)
    CPT_CPU_GPU AABB(V1Type&& _mini, V2Type&& _maxi, int _obj_idx, int _prim_idx):
        mini(std::forward<V1Type>(_mini)), __bytes1(_obj_idx), 
        maxi(std::forward<V2Type>(_maxi)), __bytes2(_prim_idx) {}

    CPT_CPU_GPU AABB(const Vec3& p1, const Vec3& p2, const Vec3& p3, int _obj_idx, int _prim_idx):
        __bytes1(_obj_idx), __bytes2(_prim_idx)
    {
        mini = p1.minimize(p2).minimize(p3);
        mini -= AABB_EPS;
        maxi = p1.maximize(p2).maximize(p3);
        maxi += AABB_EPS;
    }

    CPT_CPU_GPU Vec3 centroid() const noexcept {return (maxi + mini) * 0.5f;}
    CPT_CPU_GPU Vec3 range()    const noexcept {return maxi - mini;}

    CPT_CPU_GPU bool intersect(const Ray& ray, float& t_near) const {
        auto invDir = 1.0f / ray.d;
        // long scoreboard
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

    CPT_CPU_GPU_INLINE int obj_idx() const { return __bytes1; }
    CPT_CPU_GPU_INLINE int& obj_idx() { return __bytes1; }

    CPT_CPU_GPU_INLINE int prim_idx() const { return __bytes2; }
    CPT_CPU_GPU_INLINE int& prim_idx() { return __bytes2; }

    CPT_CPU_GPU_INLINE int base() const { return __bytes1; }
    CPT_CPU_GPU_INLINE int& base() { return __bytes1; }

    CPT_CPU_GPU_INLINE int prim_cnt() const { return __bytes2; }
    CPT_CPU_GPU_INLINE int& prim_cnt() { return __bytes2; }
};

struct AABBWrapper {
    AABB aabb;
    float4 _padding;            // padding is here to avoid bank conflict
};

using ConstAABBPtr = const AABB* const;
using ConstAABBWPtr = const AABBWrapper* const;