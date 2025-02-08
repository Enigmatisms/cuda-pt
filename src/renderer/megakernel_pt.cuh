/**
 * Megakernel Path Tracing
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include "core/bvh.cuh"
#include "bsdf/bsdf.cuh"
#include "core/max_depth.h"
#include "core/emitter.cuh"
#include "core/primitives.cuh"
#include "core/camera_model.cuh"

extern CPT_GPU_CONST Emitter* c_emitter[9];          // c_emitter[8] is a dummy emitter
extern CPT_GPU_CONST BSDF*    c_material[48];

using ConstNodePtr  = const LinearNode* const __restrict__;
using ConstObjPtr   = const CompactedObjInfo* const __restrict__;
using ConstBSDFPtr  = const BSDF* const __restrict__;
using ConstIndexPtr = const int* const __restrict__;

inline CPT_GPU_INLINE int extract_object_info(uint32_t obj_idx, bool& is_triangle) {
    is_triangle = (obj_idx & 0x80000000) == 0;
    return obj_idx & 0x000fffff;                            // extract low 20bits
}

// occlusion test is any hit shader
inline CPT_GPU bool occlusion_test_bvh(
    const Ray& ray,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const PrecomputedArray& verts,
    const int node_num,
    const int cache_num,
    float max_dist
) {
    int node_idx     = node_num;
    float aabb_tmin  = 0;
    Vec3 inv_d = ray.d.rcp(), o_div = ray.o * inv_d; 
    // There can be much control flow divergence, not good
    for (int i = 0; i < cache_num;) {
        const CompactNode node(cached_nodes[i]);
        bool intersect_node = node.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < max_dist;
        int all_offset = node.get_cached_offset(), gmem_index = node.get_gmem_index();
        int increment = (!intersect_node) * all_offset + int(intersect_node && all_offset != 1);
        // reuse
        intersect_node = intersect_node && all_offset == 1;
        i = intersect_node ? cache_num : (i + increment);
        node_idx = intersect_node ? gmem_index : node_idx;
    }
    // no intersected nodes, for the near root level, meaning that the path is not occluded
    while (node_idx < node_num) {
        const LinearNode node(tex1Dfetch<float4>(nodes, 2 * node_idx), 
                        tex1Dfetch<float4>(nodes, 2 * node_idx + 1));

        bool intersect_node = node.aabb.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < max_dist;
        int beg_idx = 0, end_idx = 0;
        node.get_range(beg_idx, end_idx);
        // Strange `increment`, huh? See the comments in function `ray_intersect_bvh`
        node_idx += (!intersect_node) * (end_idx < 0 ? -end_idx : 1) + int(intersect_node);
        end_idx = intersect_node ? end_idx + beg_idx : 0;
        for (int idx = beg_idx; idx < end_idx; idx ++) {
#ifdef TRIANGLE_ONLY
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v, true, EPSILON, max_dist);
#else
            uint32_t obj_info = tex1Dfetch<uint32_t>(bvh_leaves, idx);
            bool is_triangle = false;
            extract_object_info(obj_info, is_triangle);
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v, is_triangle, EPSILON, max_dist);
#endif
            if (dist > EPSILON)
                return false;
        }
    }
    return true;
}

/**
 * Stackless BVH (should use tetxure memory?)
 * Perform ray-intersection test on shared memory primitives
 * @param ray: the ray for intersection test
 * @param s_aabbs: scene primitive AABB
 * @param shape_visitor: encapsulated shape visitor
 * @param it: interaction info, containing the interacted normal and uv
 * @param remain_prims: number of primitives to be tested (32 at most)
 * @param cp_base: shared memory address offset
 * @param min_dist: current minimal distance
 *
 * @return minimum intersection distance
 * 
 * ray_intersect_bvh is closesthit shader
 * compare to the ray_intersect_old, this API almost double the speed
*/
inline CPT_GPU float ray_intersect_bvh(
    const Ray& ray,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const PrecomputedArray& verts,
    int& min_index,
    uint32_t& min_obj_info,
    float& prim_u,
    float& prim_v,
    const int node_num,
    const int cache_num,
    float min_dist = MAX_DIST
) {
    int node_idx     = 0;
    float aabb_tmin  = 0;
    // There can be much control flow divergence, not good
    Vec3 inv_d = ray.d.rcp(), o_div = ray.o * inv_d; 
    for (int i = 0; i < cache_num;) {
        const CompactNode node(cached_nodes[i]);
        bool intersect_node = node.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < min_dist;
        int all_offset = node.get_cached_offset(), gmem_index = node.get_gmem_index();
        int increment = (!intersect_node) * all_offset + int(intersect_node && all_offset != 1);
        // reuse
        intersect_node = intersect_node && all_offset == 1;
        i = intersect_node ? cache_num : (i + increment);
        node_idx = intersect_node ? gmem_index : node_idx;
    }
    // There can be much control flow divergence, not good
    while (node_idx < node_num) {
        const LinearNode node(tex1Dfetch<float4>(nodes, 2 * node_idx), 
                        tex1Dfetch<float4>(nodes, 2 * node_idx + 1));
        int beg_idx = 0, end_idx = 0;
        node.get_range(beg_idx, end_idx);
        bool intersect_node = node.aabb.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < min_dist;
        // The logic here: end_idx is reuse, if end_idx < 0, meaning that the current node is
        // non-leaf, non-leaf node stores (-all_offset) as end_idx, so to skip the node and its children
        // -end_idx will be the offset. While for leaf node, 1 will be the increment offset, and `POSITIVE` end_idx
        // is stored. So the following for loop can naturally run (while for non-leaf, naturally skip)
        node_idx += (!intersect_node) * (end_idx < 0 ? -end_idx : 1) + int(intersect_node);
        end_idx = intersect_node ? end_idx + beg_idx : 0;
        for (int idx = beg_idx; idx < end_idx; idx ++) {
            // if current ray intersects primitive at [idx], tasks will store it
            uint32_t obj_info = tex1Dfetch<uint32_t>(bvh_leaves, idx);
            bool is_triangle = false;
            extract_object_info(obj_info, is_triangle);
#ifdef TRIANGLE_ONLY
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v);
#else
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v, is_triangle);
#endif
            bool valid = dist > EPSILON && dist < min_dist;
            min_dist = valid ? dist : min_dist;
            prim_u   = valid ? it_u : prim_u;
            prim_v   = valid ? it_v : prim_v;
            min_index = valid ? idx : min_index;
            min_obj_info = valid ? obj_info : min_obj_info;
        }
    }
    return min_dist;
}

/**
 * @brief online variance calculation via Welford's algorithm
 * @param var_buffer    Biased variance buffer (float)
 * @param local_v       Accumulation  (time step: accum_cnt - 1)
 * @param radiance      Current input (time step: accum_cnt)
 * @param px            Image buffer index (x)
 * @param py            Image buffer index (y)
 * @param img_w         Image buffer width
 * @param accum_cnt     Current time step
 * @note The variance is the Biased sample variance
 */
inline CPT_GPU void estimate_variance(
    float* __restrict__ var_buffer,
    const Vec4& local_v,
    const Vec4& radiance,
    int px, 
    int py,
    int img_w,
    int accum_cnt
) {
    float cur_val  = (radiance.x() + radiance.y() + radiance.z()) * 0.33333333f,
          old_mean = (local_v.x() + local_v.y() + local_v.z()) * 0.33333333f,
          new_mean = old_mean + cur_val, rcp_cnt = 1.f / float(accum_cnt);
    old_mean  = accum_cnt > 1 ? old_mean / float(accum_cnt - 1) : 0;
    new_mean *= rcp_cnt;
    int index = px + py * img_w;
    // this is biased sample variance
    var_buffer[index] = (float(accum_cnt - 1) * var_buffer[index] + (cur_val - old_mean) * (cur_val - new_mean)) * rcp_cnt;
}

CPT_GPU_INLINE Emitter* sample_emitter(Sampler& sampler, float& pdf, int num, int no_sample) {
    // logic: if no_sample and num > 1, means that there is one emitter that can not be sampled
    // so we can only choose from num - 1 emitters, the following computation does this (branchless)
    // if (emit_id >= no_sample && no_sample >= 0) -> we should skip one index (the no_sample), therefore + 1
    // if invalid (there is only one emitter, and we cannot sample it), return c_emitter[8]
    // if no_sample is 0x08, then the ray hits no emitter
    num -= no_sample > 0 && num > 1;
    uint32_t emit_id = (sampler.discrete1D() % uint32_t(num)) + 1;
    emit_id += emit_id >= no_sample && no_sample > 0;
    pdf = 1.f / float(num);
    // when no_sample == 0 (means, we do not intersect any emitter) or num > 1 (there are more than 1 emitters)
    // the sample will be valid
    return c_emitter[emit_id * uint32_t(no_sample == 0 || num > 1)];
}

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level culling
 * shared memory might not be easy to use, since the memory granularity will be
 * too difficult to control
 * 
 * @param objects        object encapsulation
 * @param verts          vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms          normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs            uv coordinates, Packed 3 Half2 and 1 int for padding (sum up to 128 bits)
 * @param emitter_prims  Primitive indices for emission objects
 * @param bvh_leaves     BVH leaf nodes (int texture, storing primitive to obj index map)
 * @param nodes          BVH nodes (32 Bytes)
 * @param cached_nodes   BVH cached nodes (in shared memory): first half: front float4, second half: back float4
 * @param image          GPU image buffer
 * @param output_buffer  Possible visualization buffer
 * @param num_prims      number of primitives (to be intersected with)
 * @param num_objects    number of objects
 * @param num_emitter    number of emitters
 * @param seed_offset    offset to random seed (to create uncorrelated samples)
 * @param md_params      maximum allowed bounces (total, diffuse, specular, transmission)
 * @param node_num       number of nodes on a BVH tree
 * @param accum_cnt      Counter of iterations
 * @param cache_num      Number of cached BVH nodes
 * @param gamma_corr     For online rendering, whether to enable gamma correction on visualization
*/
template <bool render_once>
CPT_KERNEL void render_pt_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const NormalArray norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    const MaxDepthParams md_params,
    float* __restrict__ output_buffer,
    float* __restrict__ var_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int node_num  = -1,
    int accum_cnt = 1,
    int cache_num = 0,
    int envmap_id = 0,
    bool gamma_corr = false
);

/**
 * Megakernel Light Tracing. Light tracing is only used to render
 * complex caustics: starting from the emitter, we will only record
 * path which has more than `specular_constraints` number of
 * specular nodes
 * @param specular_constraints The path throughput will be ignored
 * if number of specular events is less or equal to this value
*/
template <bool render_once>
CPT_KERNEL void render_lt_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const NormalArray norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    const MaxDepthParams md_params,
    float* __restrict__ output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int node_num = -1,
    int accum_cnt = 1,
    int cache_num = 0,
    int specular_constraints = 0,
    float caustic_scale = 1.f,
    bool gamma_corr = false
);