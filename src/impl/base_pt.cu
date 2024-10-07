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
) {
    float aabb_tmin = 0; 
    #pragma unroll
    for (int idx = 0; idx < remain_prims; idx ++) {
        auto aabb = s_aabbs[idx].aabb;
        if (aabb.intersect(ray, aabb_tmin) && aabb_tmin < min_dist) {
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, s_verts, idx, it_u, it_v, aabb.obj_idx() >= 0);
            bool valid = dist > EPSILON && dist < min_dist;
            min_dist = valid ? dist : min_dist;
            prim_u   = valid ? it_u : prim_u;
            prim_v   = valid ? it_v : prim_v;
            min_index = valid ? cp_base + idx : min_index;
            min_obj_idx = valid ? aabb.obj_idx() : min_obj_idx;
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
        int obj_idx = s_aabbs[idx].aabb.obj_idx();
#ifdef TRIANGLE_ONLY
        float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, s_verts, idx, it_u, it_v, true);
#else
        float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, s_verts, idx, it_u, it_v, obj_idx >= 0);
#endif
        bool valid = dist > EPSILON && dist < min_dist;
        min_dist = valid ? dist : min_dist;
        prim_u   = valid ? it_u : prim_u;
        prim_v   = valid ? it_v : prim_v;
        min_index = valid ? cp_base + idx : min_index;
        min_obj_idx = valid ? obj_idx : min_obj_idx;
    }
    return min_dist;
}