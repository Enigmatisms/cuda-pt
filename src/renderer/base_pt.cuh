/**
 * Base class of path tracers
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/shapes.cuh"

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
    const Ray& ray,
    ConstShapePtr shapes,
    ConstAABBWPtr s_aabbs,
    ShapeIntersectVisitor& shape_visitor,
    int& min_index,
    const int remain_prims,
    const int cp_base,
    float min_dist
);

/**
 * Perform ray-intersection test on shared memory primitives
 * @param ray: the ray for intersection test
 * @param shapes: scene primitives
 * @param s_aabbs: scene primitives
 * @param shape_visitor: encapsulated shape visitor
 * @param it: interaction info, containing the interacted normal and uv
 * @param remain_prims: number of primitives to be tested (32 at most)
 * @param cp_base: shared memory address offset
 * @param min_dist: current minimal distance
 *
 * @return minimum intersection distance
 * 
 * compare to the ray_intersect_old, this API almost double the speed
*/
CPT_GPU float ray_intersect(
    const Ray& ray,
    ConstShapePtr shapes,
    ConstAABBWPtr s_aabbs,
    ShapeIntersectVisitor& shape_visitor,
    int& min_index,
    const int remain_prims,
    const int cp_base,
    float min_dist
);