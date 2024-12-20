/**
 * Megakernel Path Tracing (Implementation)
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#include "renderer/base_pt.cuh"
#include "renderer/megakernel_pt.cuh"

static constexpr int RR_BOUNCE = 2;
static constexpr float RR_THRESHOLD = 0.1;

/**
 * Occlusion test, computation is done on global memory
*/
CPT_GPU bool occlusion_test(
    const Ray& ray,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    const PrecomputedArray& verts,
    int num_objects,
    float max_dist
) {
    float aabb_tmin = 0;
    int prim_id = 0, num_prim = 0;
    for (int obj_id = 0; obj_id < num_objects; obj_id ++) {
        num_prim = prim_id + objects[obj_id].prim_num;
        if (objects[obj_id].intersect(ray, aabb_tmin) && aabb_tmin < max_dist) {
            // ray intersects the object
            for (; prim_id < num_prim; prim_id ++) {
                auto aabb = aabbs[prim_id];
                if (aabb.intersect(ray, aabb_tmin) && aabb_tmin < max_dist) {
                    float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, prim_id, it_u, it_v, aabb.obj_idx() >= 0);
                    if (dist < max_dist && dist > EPSILON)
                        return false;
                }
            }
        }
        prim_id = num_prim;
    }
    return true;
}

// occlusion test is any hit shader
CPT_GPU bool occlusion_test_bvh(
    const Ray& ray,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const PrecomputedArray& verts,
    const int node_num,
    const int cache_num,
    float max_dist
) {
    bool valid_cache = false;
    int node_idx     = 0;
    float aabb_tmin  = 0;
    Vec3 inv_d = ray.d.rcp(), o_div = ray.o * inv_d; 
    // There can be much control flow divergence, not good
    while (node_idx < cache_num && !valid_cache) {
        const LinearNode node(
            cached_nodes[node_idx],
            cached_nodes[node_idx + cache_num]
        );
        bool intersect_node = node.aabb.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < max_dist;
        int all_offset = node.aabb.base(), gmem_index = node.aabb.prim_cnt();
        int increment = (!intersect_node) * all_offset + int(intersect_node && all_offset != 1);
        node_idx += increment;

        if (intersect_node && all_offset == 1) {
            valid_cache = true;
            node_idx = gmem_index;
        }
    }
    // no intersected nodes, for the near root level, meaning that the path is not occluded
    if (valid_cache) {
        while (node_idx < node_num) {
            const LinearNode node(tex1Dfetch<float4>(nodes, 2 * node_idx), 
                            tex1Dfetch<float4>(nodes, 2 * node_idx + 1));

            bool intersect_node = node.aabb.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < max_dist;
            int beg_idx = 0, end_idx = 0;
            node.get_range(beg_idx, end_idx);
            // Strange `increment`, huh? See the comments in function `ray_intersect_bvh`
            int increment = (!intersect_node) * (end_idx < 0 ? -end_idx : 1) + int(intersect_node);
            if (intersect_node && end_idx > 0) {
                end_idx += beg_idx;
                for (int idx = beg_idx; idx < end_idx; idx ++) {
                    // if current ray intersects primitive at [idx], tasks will store it
                    int obj_idx = tex1Dfetch<int>(bvh_leaves, idx);
                    float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v, obj_idx >= 0);
                    if (dist > EPSILON && dist < max_dist)
                        return false;
                }
            }
            node_idx += increment;
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
CPT_GPU float ray_intersect_bvh(
    const Ray& ray,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const PrecomputedArray& verts,
    int& min_index,
    int& min_obj_idx,
    float& prim_u,
    float& prim_v,
    const int node_num,
    const int cache_num,
    float min_dist
) {
    bool valid_cache = false;
    int node_idx     = 0;
    float aabb_tmin  = 0;
    // There can be much control flow divergence, not good
    Vec3 inv_d = ray.d.rcp(), o_div = ray.o * inv_d; 
    while (node_idx < cache_num && !valid_cache) {
        const LinearNode node(
            cached_nodes[node_idx],
            cached_nodes[node_idx + cache_num]
        );
        bool intersect_node = node.aabb.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < min_dist;
        int all_offset = node.aabb.base(), gmem_index = node.aabb.prim_cnt();
        int increment = (!intersect_node) * all_offset + (intersect_node && all_offset != 1) * 1;

        node_idx += increment;

        if (intersect_node && all_offset == 1) {
            valid_cache = true;
            node_idx = gmem_index;
        }
    }
    if (valid_cache) {
        // There can be much control flow divergence, not good
        while (node_idx < node_num) {
            const LinearNode node(tex1Dfetch<float4>(nodes, 2 * node_idx), 
                            tex1Dfetch<float4>(nodes, 2 * node_idx + 1));
            bool intersect_node = node.aabb.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < min_dist;
            int beg_idx = 0, end_idx = 0;
            node.get_range(beg_idx, end_idx);
            // The logic here: end_idx is reuse, if end_idx < 0, meaning that the current node is
            // non-leaf, non-leaf node stores (-all_offset) as end_idx, so to skip the node and its children
            // -end_idx will be the offset. While for leaf node, 1 will be the increment offset, and `POSITIVE` end_idx
            // is stored. So the following for loop can naturally run (while for non-leaf, naturally skip)
            int increment = (!intersect_node) * (end_idx < 0 ? -end_idx : 1) + int(intersect_node);
            if (intersect_node && end_idx > 0) {
                end_idx += beg_idx;
                for (int idx = beg_idx; idx < end_idx; idx ++) {
                    // if current ray intersects primitive at [idx], tasks will store it
                    bool valid  = false;
                    int obj_idx = tex1Dfetch<int>(bvh_leaves, idx);
                    float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v, obj_idx >= 0);
                    valid = dist > EPSILON && dist < min_dist;
                    min_dist = valid ? dist : min_dist;
                    prim_u   = valid ? it_u : prim_u;
                    prim_v   = valid ? it_v : prim_v;
                    min_index = valid ? idx : min_index;
                    min_obj_idx = valid ? obj_idx : min_obj_idx;
                }
            }
            node_idx += increment;
        }
    }
    return min_dist;
}

template <bool render_once>
CPT_KERNEL void render_pt_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int node_num,
    int accum_cnt,
    int cache_num,
    bool gamma_corr
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;

    Sampler sampler(px + py * image.w(), seed_offset);
    // step 1: generate ray

    // step 2: bouncing around the scene until the max depth is reached
    int min_index = -1, object_id = 0;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    extern __shared__ float4 s_cached[];
#ifdef RENDERER_USE_BVH
    // cache near root level BVH nodes for faster traversal
    if (tid < 2 * cache_num) {      // no more than 128 nodes will be cached
        s_cached[tid] = cached_nodes[tid];
    }
    Ray ray = dev_cam.generate_ray(px, py, sampler);
    __syncthreads();
#else
    Ray ray = dev_cam.generate_ray(px, py, sampler);
    __shared__ Vec4 s_verts[TRI_IDX(BASE_ADDR)];         // vertex info
    __shared__ AABBWrapper s_aabbs[BASE_ADDR];            // aabb
    PrecomputedArray s_verts_arr(reinterpret_cast<Vec4*>(&s_verts[0]), BASE_ADDR);
    int num_copy = (num_prims + BASE_ADDR - 1) / BASE_ADDR;   // round up
#endif  // RENDERER_USE_BVH

    Vec4 throughput(1, 1, 1), radiance(0, 0, 0);
    float emission_weight = 1.f;
    bool hit_emitter = false;
    
    for (int b = 0; b < max_depth; b++) {
        float prim_u = 0, prim_v = 0, min_dist = MAX_DIST;
        min_index = -1;
        // ============= step 1: ray intersection =================
#ifdef RENDERER_USE_BVH
        min_dist = ray_intersect_bvh(
            ray, bvh_leaves, nodes, s_cached, 
            verts, min_index, object_id, 
            prim_u, prim_v, node_num, cache_num, min_dist
        );
#else   // RENDERER_USE_BVH
        #pragma unroll
        for (int cp_base = 0; cp_base < num_copy; ++cp_base) {
            // memory copy to shared memory
            int cur_idx = (cp_base << BASE_SHFL) + tid, remain_prims = min(num_prims - (cp_base << BASE_SHFL), BASE_ADDR);
            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

            if (tid < BASE_ADDR && cur_idx < num_prims) {        // copy from gmem to smem
                cuda::memcpy_async(&s_verts[TRI_IDX(tid)], &verts.data[TRI_IDX(cur_idx)], sizeof(Vec4) * 3, pipe);
                s_aabbs[tid].aabb.copy_from(aabbs[cur_idx]);
            }
            pipe.producer_commit();
            pipe.consumer_wait();
            __syncthreads();
            // this might not be a good solution
            min_dist = ray_intersect(s_verts_arr, ray, s_aabbs, remain_prims, 
                cp_base << BASE_SHFL, min_index, object_id, prim_u, prim_v, min_dist);
            __syncthreads();
        }
#endif  // RENDERER_USE_BVH
        // ============= step 2: local shading for indirect bounces ================
        if (min_index >= 0) {
            auto it = Primitive::get_interaction(verts, norms, uvs, ray.advance(min_dist), prim_u, prim_v, min_index, object_id >= 0);
            object_id = object_id >= 0 ? object_id : -object_id - 1;        // sphere object ID is -id - 1

            // ============= step 3: next event estimation ================
            // (1) randomly pick one emitter
            int material_id = objects[object_id].bsdf_id,
                emitter_id  = objects[object_id].emitter_id;
            hit_emitter = emitter_id > 0;

            // emitter MIS
            emission_weight = emission_weight / (emission_weight + 
                    objects[object_id].solid_angle_pdf(it.shading_norm, ray.d, min_dist) * 
                    hit_emitter * (b > 0) * ray.non_delta());

            float direct_pdf = 1;       // direct_pdf is the product of light_sampling_pdf and emitter_pdf
            // (2) check if the ray hits an emitter
            Vec4 direct_comp = throughput *\
                        c_emitter[emitter_id]->eval_le(&ray.d, &it.shading_norm);
            radiance += direct_comp * emission_weight;

            const Emitter* emitter = sample_emitter(sampler, c_emitter, direct_pdf, num_emitter, emitter_id);
            // (3) sample a point on the emitter (we avoid sampling the hit emitter)
            emitter_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(sampler.discrete1D(), direct_pdf);
            emitter_id = emitter_prims[emitter_id];               // extra mapping, introduced after BVH primitive reordering
            Ray shadow_ray(ray.advance(min_dist), Vec3(0, 0, 0));
            // use ray.o to avoid creating another shadow_int variable
            shadow_ray.d = emitter->sample(shadow_ray.o, direct_comp, direct_pdf, sampler.next2D(), verts, norms, emitter_id) - shadow_ray.o;
            
            float emit_len_mis = shadow_ray.d.length();
            shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direction

            // (3) NEE scene intersection test (possible warp divergence, but... nevermind)
            if (emitter != c_emitter[0] &&
#ifdef RENDERER_USE_BVH
                occlusion_test_bvh(shadow_ray, bvh_leaves, nodes, s_cached, 
                            verts, node_num, cache_num, emit_len_mis - EPSILON)
#else   // RENDERER_USE_BVH
                occlusion_test(shadow_ray, objects, aabbs, verts, num_objects, emit_len_mis - EPSILON)
#endif  // RENDERER_USE_BVH
            ) {
                // MIS for BSDF / light sampling, to achieve better rendering
                // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
                emit_len_mis = direct_pdf + c_material[material_id]->pdf(it, shadow_ray.d, ray.d) * emitter->non_delta();
                radiance += throughput * direct_comp * c_material[material_id]->eval(it, shadow_ray.d, ray.d) * \
                    (float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
                // numerical guard, in case emit_len_mis is 0
            }

            // step 4: sample a new ray direction, bounce the 
            BSDFFlag sampled_lobe = BSDFFlag::BSDF_NONE;
            ray.o = std::move(shadow_ray.o);
            ray.d = c_material[material_id]->sample_dir(ray.d, it, throughput, emission_weight, sampler, sampled_lobe);
            ray.set_delta((BSDFFlag::BSDF_SPECULAR & sampled_lobe) > 0);

            if (radiance.numeric_err())
                radiance.fill(0);
            
#ifdef RENDERER_USE_BVH
            // using BVH enables the usage of RR, since there is no within-loop synchronization
            float max_value = throughput.max_elem_3d();
            if (b >= RR_BOUNCE && max_value < RR_THRESHOLD) {
                if (sampler.next1D() > max_value || max_value < THP_EPS) break;
                throughput *= 1. / max_value;
            }
#endif // RENDERER_USE_BVH
        }
    }
    if constexpr (render_once) {
        // image will be the output buffer, there will be double buffering
        auto local_v = image(px, py) + radiance;
        image(px, py) = local_v;
        local_v *= 1.f / float(accum_cnt);
        local_v = gamma_corr ? local_v.gamma_corr() : local_v;
        FLOAT4(output_buffer[(px + py * image.w()) << 2]) = float4(local_v); 
    } else {
        image(px, py) += radiance;
    }
}

template CPT_KERNEL void render_pt_kernel<true>(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int node_num,
    int accum_cnt,
    int cache_num,
    bool gamma_corr
);

template CPT_KERNEL void render_pt_kernel<false>(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int node_num,
    int accum_cnt,
    int cache_num,
    bool gamma_corr
);