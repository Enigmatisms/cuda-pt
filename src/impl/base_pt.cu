/**
 * Base class of path tracers (implementation)
 * @date: 9.16.2024
 * @author: Qianyue He
*/
#include "renderer/base_pt.cuh"

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
) {
    float aabb_tmin = 0; 
    #pragma unroll
    for (int idx = 0; idx < remain_prims; idx ++) {
        if (s_aabbs[idx].aabb.intersect(ray, aabb_tmin) && aabb_tmin < min_dist) {
            shape_visitor.set_index(idx);
            float dist = variant::apply_visitor(shape_visitor, shapes[cp_base + idx]);
            bool valid = dist > EPSILON && dist < min_dist;
            min_dist = valid ? dist : min_dist;
            min_index = valid ? cp_base + idx : min_index;
        }
    }
    return min_dist;
}

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
) {
    float aabb_tmin = 0;
    BitMask tasks = 0;

#pragma unroll
    for (int idx = 0; idx < remain_prims; idx ++) {
        // if current ray intersects primitive at [idx], tasks will store it
        BitMask valid_intr = s_aabbs[idx].aabb.intersect(ray, aabb_tmin) && aabb_tmin < min_dist;
        tasks |= valid_intr << (BitMask)idx;
    }
#pragma unroll
    while (tasks) {
        BitMask idx = __count_bit(tasks) - 1; // find the first bit that is set to 1, note that __ffs is 
        tasks &= ~((BitMask)1 << idx); // clear bit in case it is found again
        shape_visitor.set_index(idx);
        float dist = variant::apply_visitor(shape_visitor, shapes[cp_base + idx]);
        bool valid = dist > EPSILON && dist < min_dist;
        min_dist = valid ? dist : min_dist;
        min_index = valid ? cp_base + idx : min_index;
    }
    return min_dist;
}