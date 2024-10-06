/**
 * Base class of path tracers
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/primitives.cuh"
#include "core/aabb.cuh"

// #define CP_BASE_6
#ifdef CP_BASE_6
constexpr int BASE_SHFL = 6;
using BitMask = long long;
CPT_GPU_INLINE int __count_bit(BitMask bits) { return __ffsll(bits); } 
#else
constexpr int BASE_SHFL = 5;
using BitMask = int;
CPT_GPU_INLINE int __count_bit(BitMask bits) { return __ffs(bits); } 
#endif
constexpr int BASE_ADDR = 1 << BASE_SHFL;

/**
 * This API is deprecated, due to the performance bounded by BSYNC
 * which is the if branch barrier synchronization (convergence problem)
 * 
 * Take a look at the stackoverflow post I posted:
 * https://stackoverflow.com/questions/78603442/convergence-barrier-for-branchless-cuda-conditional-select
*/
CPT_GPU float ray_intersect_old(
    const ArrayType<Vec3>& s_verts, 
    const Ray& ray,
    ConstAABBWPtr s_aabbs,
    const int remain_prims,
    const int cp_base,
    int& min_index,
    int& min_obj_idx,
    float& prim_u,
    float& prim_v,
    float min_dist
);

/**
 * Perform ray-intersection test on shared memory primitives
 * @param s_verts:      vertices stored in shared memory
 * @param ray:          the ray for intersection test
 * @param s_aabbs:      scene primitives
 * @param remain_prims: number of primitives to be tested (32 at most)
 * @param min_index:    closest hit primitive index
 * @param min_obj_idx:  closest hit object index
 * @param prim_u:       intersection barycentric coord u
 * @param prim_v:       intersection barycentric coord v
 * @param min_dist:     current minimal distance
 *
 * @return minimum intersection distance
 * 
 * compare to the ray_intersect_old, this API almost double the speed
*/
CPT_GPU float ray_intersect(
    const ArrayType<Vec3>& s_verts, 
    const Ray& ray,
    ConstAABBWPtr s_aabbs,
    const int remain_prims,
    const int cp_base,
    int& min_index,
    int& min_obj_idx,
    float& prim_u,
    float& prim_v,
    float min_dist
);