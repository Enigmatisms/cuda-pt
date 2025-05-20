// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

#pragma once
/**
 * @file tracing_func.cuh
 * @author Qianyue He
 * @brief Important Path Tracing Functions common to multiple tracers
 * @version 0.1
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 */

#include "bsdf/bsdf.cuh"
#include "core/bvh.cuh"
#include "core/emitter.cuh"
#include "core/primitives.cuh"

extern CPT_GPU_CONST Emitter *c_emitter[9]; // c_emitter[8] is a dummy emitter
extern CPT_GPU_CONST BSDF *c_material[48];

using ConstNodePtr = const LinearNode *const __restrict__;
using ConstObjPtr = const CompactedObjInfo *const __restrict__;
using ConstBSDFPtr = const BSDF *const __restrict__;
using ConstIndexPtr = const int *const __restrict__;

CPT_GPU_INLINE int extract_object_info(int obj_idx, bool &is_triangle) {
    is_triangle = (obj_idx & 0x80000000) == 0;
    return obj_idx & 0x000fffff; // extract low 20bits
}

// occlusion test is any hit shader
inline CPT_GPU bool
occlusion_test_bvh(const Ray &ray, const cudaTextureObject_t bvh_leaves,
                   const cudaTextureObject_t nodes, ConstF4Ptr cached_nodes,
                   const PrecomputedArray &verts, const int node_num,
                   const int cache_num, float max_dist) {
    int node_idx = node_num;
    float aabb_tmin = 0;
    Vec3 inv_d = ray.d.rcp(), o_div = ray.o * inv_d;
    // There can be much control flow divergence, not good
    for (int i = 0; i < cache_num;) {
        const CompactNode node(cached_nodes[i]);
        bool intersect_node =
            node.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < max_dist;
        int all_offset = node.get_cached_offset(),
            gmem_index = node.get_gmem_index();
        int increment = (!intersect_node) * all_offset +
                        int(intersect_node && all_offset != 1);
        // reuse
        intersect_node = intersect_node && all_offset == 1;
        i = intersect_node ? cache_num : (i + increment);
        node_idx = intersect_node ? gmem_index : node_idx;
    }
    // no intersected nodes, for the near root level, meaning that the path is
    // not occluded
    while (node_idx < node_num) {
        const LinearNode node(tex1Dfetch<float4>(nodes, 2 * node_idx),
                              tex1Dfetch<float4>(nodes, 2 * node_idx + 1));

        bool intersect_node = node.aabb.intersect(inv_d, o_div, aabb_tmin) &&
                              aabb_tmin < max_dist;
        int beg_idx = 0, end_idx = 0;
        node.get_range(beg_idx, end_idx);
        // Strange `increment`, huh? See the comments in function
        // `ray_intersect_bvh`
        node_idx += (!intersect_node) * (end_idx < 0 ? -end_idx : 1) +
                    int(intersect_node);
        end_idx = intersect_node ? end_idx + beg_idx : 0;
        for (int idx = beg_idx; idx < end_idx; idx++) {
#ifdef TRIANGLE_ONLY
            float it_u = 0, it_v = 0,
                  dist = Primitive::intersect(ray, verts, idx, it_u, it_v, true,
                                              EPSILON, max_dist);
#else
            uint32_t obj_info = tex1Dfetch<uint32_t>(bvh_leaves, idx);
            bool is_triangle = false;
            extract_object_info(obj_info, is_triangle);
            float it_u = 0, it_v = 0,
                  dist = Primitive::intersect(ray, verts, idx, it_u, it_v,
                                              is_triangle, EPSILON, max_dist);
#endif
            if (dist > EPSILON)
                return false;
        }
    }
    return true;
}

/**
 * Stackless BVH (should use texture memory?)
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
inline CPT_GPU float
ray_intersect_bvh(const Ray &ray, const cudaTextureObject_t bvh_leaves,
                  const cudaTextureObject_t nodes, ConstF4Ptr cached_nodes,
                  const PrecomputedArray &verts, int &min_index,
                  int &min_obj_info, float &prim_u, float &prim_v,
                  const int node_num, const int cache_num,
                  float min_dist = MAX_DIST) {
    int node_idx = 0;
    float aabb_tmin = 0;
    // There can be much control flow divergence, not good
    Vec3 inv_d = ray.d.rcp(), o_div = ray.o * inv_d;
    for (int i = 0; i < cache_num;) {
        const CompactNode node(cached_nodes[i]);
        bool intersect_node =
            node.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < min_dist;
        int all_offset = node.get_cached_offset(),
            gmem_index = node.get_gmem_index();
        int increment = (!intersect_node) * all_offset +
                        int(intersect_node && all_offset != 1);
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
        bool intersect_node = node.aabb.intersect(inv_d, o_div, aabb_tmin) &&
                              aabb_tmin < min_dist;
        // The logic here: end_idx is reuse, if end_idx < 0, meaning that the
        // current node is non-leaf, non-leaf node stores (-all_offset) as
        // end_idx, so to skip the node and its children -end_idx will be the
        // offset. While for leaf node, 1 will be the increment offset, and
        // `POSITIVE` end_idx is stored. So the following for loop can naturally
        // run (while for non-leaf, naturally skip)
        node_idx += (!intersect_node) * (end_idx < 0 ? -end_idx : 1) +
                    int(intersect_node);
        end_idx = intersect_node ? end_idx + beg_idx : 0;
        for (int idx = beg_idx; idx < end_idx; idx++) {
            // if current ray intersects primitive at [idx], tasks will store it
            int obj_info = tex1Dfetch<int>(bvh_leaves, idx);
#ifdef TRIANGLE_ONLY
            float it_u = 0, it_v = 0,
                  dist = Primitive::intersect(ray, verts, idx, it_u, it_v);
#else
            bool is_triangle = false;
            extract_object_info(obj_info, is_triangle);
            float it_u = 0, it_v = 0,
                  dist = Primitive::intersect(ray, verts, idx, it_u, it_v,
                                              is_triangle);
#endif
            bool valid = dist > EPSILON && dist < min_dist;
            min_dist = valid ? dist : min_dist;
            prim_u = valid ? it_u : prim_u;
            prim_v = valid ? it_v : prim_v;
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
inline CPT_GPU void estimate_variance(float *__restrict__ var_buffer,
                                      const Vec4 &local_v, const Vec4 &radiance,
                                      int px, int py, int img_w,
                                      int accum_cnt) {
    float cur_val = (radiance.x() + radiance.y() + radiance.z()) * 0.33333333f,
          old_mean = (local_v.x() + local_v.y() + local_v.z()) * 0.33333333f,
          new_mean = old_mean + cur_val, rcp_cnt = 1.f / float(accum_cnt);
    old_mean = accum_cnt > 1 ? old_mean / float(accum_cnt - 1) : 0;
    new_mean *= rcp_cnt;
    int index = px + py * img_w;
    // this is biased sample variance
    var_buffer[index] = (float(accum_cnt - 1) * var_buffer[index] +
                         (cur_val - old_mean) * (cur_val - new_mean)) *
                        rcp_cnt;
}

CPT_GPU_INLINE Emitter *sample_emitter(Sampler &sampler, float &pdf, int num,
                                       int no_sample) {
    // logic: if no_sample and num > 1, means that there is one emitter that can
    // not be sampled so we can only choose from num - 1 emitters, the following
    // computation does this (branchless) if (emit_id >= no_sample && no_sample
    // >= 0) -> we should skip one index (the no_sample), therefore + 1 if
    // invalid (there is only one emitter, and we cannot sample it), return
    // c_emitter[8] if no_sample is 0x08, then the ray hits no emitter
    num -= no_sample > 0 && num > 1;
    uint32_t emit_id = (sampler.discrete1D() % uint32_t(num)) + 1;
    emit_id += emit_id >= no_sample && no_sample > 0;
    pdf = 1.f / float(num);
    // when no_sample == 0 (means, we do not intersect any emitter) or num > 1
    // (there are more than 1 emitters) the sample will be valid
    return c_emitter[emit_id * uint32_t(no_sample == 0 || num > 1)];
}
