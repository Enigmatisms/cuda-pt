/**
 * Megakernel Volumetric Path Tracing
 * @date: 2025.2.7
 * @author: Qianyue He
 * 
 * Note that, the current implementation does not support nested volume that are 
 * 4 or more levels. [volume-1  [volume-2 [volume-3]   ]   ]  is the limit
 * Actually, it won't trigger any fault but the rendering result will not be correct
 * for more layers than 3
 * 
 * Also, non-strict nesting ( [vol-1    [vol-1 & vol-2 intersection]    vol-2]) will
 * also be erroneous, it won't break down but the result will also be incorrect.
*/
#pragma once
#include <cuda/pipeline>
#include "core/medium.cuh"
#include "renderer/megakernel_pt.cuh"

// returns medium_index and whether the object is alpha masked
inline CPT_GPU_INLINE int extract_medium_info(uint32_t obj_idx, bool& alpha_mask) {
    alpha_mask  = (obj_idx & 0x40000000) == 1;              // bit 31
     // extract higher 12 bits and mask the resulting lower 8bits
    return (obj_idx >> 20) & 0x000000ff;
}

inline CPT_GPU_INLINE int extract_tracing_info(uint32_t obj_idx, int& hit_med_idx, bool& is_triangle) {
    is_triangle = (obj_idx & 0x80000000) == 0;
     // extract higher 12 bits and mask the resulting lower 8bits
    hit_med_idx = (obj_idx >> 20) & 0x000000ff;
    return obj_idx & 0x000fffff;                            // extract low 20bits, return the object index
}

/**
 * @brief Stack with only one bank (4B), used for handling nested volumes
 * x is the ptr, if x == 0, it means that active volume is 0 (not within a volume)
 */
struct BankStack {
    uchar4 data;

    CPT_GPU_INLINE BankStack(int val = 0): data{0, 0, 0, 0} {
        if (val > 0) {
            data.x = 1;
            data.y = uint8_t(val);
        }
    }

    CPT_GPU_INLINE int top() const {
        auto ptr = &data.x;
        return data.x > 0 ? *((&data.x) + data.x) : 0;
    }

    CPT_GPU_INLINE void push(uint8_t val) {
        if (data.x < 3) {
            data.x ++;
            *((&data.x) + data.x) = val;
        }
    }

    CPT_GPU_INLINE int pop() {
        int res = 0;
        if (data.x > 0) {
            res = *((&data.x) + data.x);
            data.x --;
        }
        return res;
    }
};

// non-binary version of occlusion test, different in 3 aspects:
// (1) Will check the obj_idx. Now, obj_idx is masked: the lower 20 bits represents
// object index, while the higher 12 bits are flag bits, if ALPHA flag is set true, 
// then during this phase, the occlusion will be ignored
// (2) Will try to accumulate the transmittance along the path. This entails fetching
// the material and medium info from the GMEM so this kernel will be much slower
// (3) The occlusion_test is replaced by closest_hit shader, since we will need 
// to step through the scenes
inline CPT_GPU Vec4 occlusion_transmittance_estimate(
    Ray ray,
    Sampler& sp,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const PrecomputedArray& verts,
    const NormalArray norms, 
    MediumPtrArray media,
    const int node_num,
    const int cache_num,
    BankStack nested_vols,
    float max_dist
) {
    float total_dist = 0;
    Vec3 inv_d = ray.d.rcp();
    Vec4 Tr(1);
    while (total_dist + EPSILON < max_dist) {
        int node_idx     = 0, min_index = -1;
        float aabb_tmin  = 0, prim_u = 0, prim_v = 0, min_dist = max_dist - total_dist;     // FIXME: precision problem
        uint32_t min_obj_info = INVALID_OBJ;
        // There can be much control flow divergence, not good
        Vec3 o_div = ray.o * inv_d;         // FIXME
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
                bool is_triangle = true;
                uint32_t obj_info = tex1Dfetch<int>(bvh_leaves, idx);
                int obj_idx = extract_object_info(obj_info, is_triangle);
#ifdef TRIANGLE_ONLY
                float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v);
#else
                float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v, is_triangle);
#endif
                bool valid = dist > EPSILON && dist < min_dist;
                min_dist = valid ? dist : min_dist;
                prim_u   = valid ? it_u : prim_u;
                prim_v   = valid ? it_v : prim_v;
                min_index    = valid ? idx : min_index;
                min_obj_info = valid ? obj_info : min_obj_info;
            }
        }
        bool is_alpha_mask = false;
        int active_medium = extract_medium_info(min_obj_info, is_alpha_mask);
        if (active_medium == 0 || is_alpha_mask == false) {
            // object with no volume binding and can not be culled (alpha mode) -> occlusion
            Tr.fill(0);
            break;
        }
        // if hit within range, and the normal is in the same hemisphere as that of the ray direction
        // This will mean that we are penetraing out from a translucent bound
        bool is_in_medium = min_index >= 0 && norms.eval(max(min_index, 0), prim_u, prim_v).dot(ray.d) > 0;
        // Oh, I hate branches in CUDA
        if (is_in_medium) {
            active_medium = nested_vols.pop();
        } else {
            active_medium = nested_vols.top();
            nested_vols.push(active_medium);
        }
        Tr *= media[active_medium]->transmittance(ray, sp, min_dist);
        
        total_dist += min_dist;
        ray.o = ray.advance(min_dist);
    }
    return Tr;
}

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level culling
 * shared memory might not be easy to use, since the memory granularity will be
 * too difficult to control
 * 
 * @param verts          vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms          normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs            uv coordinates, Packed 3 Half2 and 1 int for padding (sum up to 128 bits)
 * @param objects        object encapsulation
 * @param media          Array of Medium base class pointers
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
 * @param cam_vol_idx    If camera is inside the volume, the ray spawned will have initial volume id to store
 * @param md_params      maximum allowed bounces (total, diffuse, specular, transmission) 
 * @param node_num       number of nodes on a BVH tree
 * @param accum_cnt      Counter of iterations
 * @param cache_num      Number of cached BVH nodes
 * @param gamma_corr     For online rendering, whether to enable gamma correction on visualization
*/
template <bool render_once>
CPT_KERNEL void render_vpt_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const NormalArray norms, 
    const ConstBuffer<PackedHalf2> uvs,
    MediumPtrArray media,
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
    int cam_vol_idx = 0,
    int node_num  = -1,
    int accum_cnt = 1,
    int cache_num = 0,
    int envmap_id = 0,
    bool gamma_corr = false
);