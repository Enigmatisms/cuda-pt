/**
 * Megakernel Path Tracing (Implementation)
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#include "core/textures.cuh"
#include "renderer/megakernel_pt.cuh"

static constexpr int RR_BOUNCE = 2;
static constexpr float RR_THRESHOLD = 0.1;

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
            int obj_idx = tex1Dfetch<int>(bvh_leaves, idx);
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v, obj_idx >= 0, EPSILON, max_dist);
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
            int obj_idx = tex1Dfetch<int>(bvh_leaves, idx);
#ifdef TRIANGLE_ONLY
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v);
#else
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v, obj_idx >= 0);
#endif
            bool valid = dist > EPSILON && dist < min_dist;
            min_dist = valid ? dist : min_dist;
            prim_u   = valid ? it_u : prim_u;
            prim_v   = valid ? it_v : prim_v;
            min_index = valid ? idx : min_index;
            min_obj_idx = valid ? obj_idx : min_obj_idx;
        }
    }
    return min_dist;
}

CPT_GPU Emitter* sample_emitter(Sampler& sampler, float& pdf, int num, int no_sample) {
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
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int node_num,
    int accum_cnt,
    int cache_num,
    int envmap_id,
    bool gamma_corr
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;

    Sampler sampler(px + py * image.w(), seed_offset);
    // step 1: generate ray

    // step 2: bouncing around the scene until the max depth is reached
    int min_index = -1, object_id = 0, diff_b = 0, spec_b = 0, trans_b = 0;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    extern __shared__ uint4 s_cached[];
    // cache near root level BVH nodes for faster traversal
    if (tid < cache_num) {      // no more than 256 nodes will be cached
        s_cached[tid] = cached_nodes[tid];
        int offset_tid = tid + blockDim.x * blockDim.y;
        if (offset_tid < cache_num)
            s_cached[offset_tid] = cached_nodes[offset_tid];
    }
    Ray ray = dev_cam.generate_ray(px, py, sampler);
    __syncthreads();

    Vec4 throughput(1, 1, 1), radiance(0, 0, 0);
    float emission_weight = 1.f;
    bool hit_emitter = false;
    
    for (int b = 0; b < md_params.max_depth; b++) {
        float prim_u = 0, prim_v = 0, min_dist = MAX_DIST;
        min_index = -1;
        // ============= step 1: ray intersection =================
        min_dist = ray_intersect_bvh(
            ray, bvh_leaves, nodes, s_cached, 
            verts, min_index, object_id, 
            prim_u, prim_v, node_num, cache_num, min_dist
        );

        // ============= step 2: local shading for indirect bounces ================
        if (min_index >= 0) {
            auto it = Primitive::get_interaction(verts, norms, uvs, ray.advance(min_dist), prim_u, prim_v, min_index, object_id >= 0);
            object_id = object_id >= 0 ? object_id : -object_id - 1;        // sphere object ID is -id - 1

            // ============= step 3: next event estimation ================
            // (1) randomly pick one emitter
            int material_id = 0, emitter_id = -1;
            objects[object_id].unpack(material_id, emitter_id);
            hit_emitter = emitter_id > 0;

            // emitter MIS
            emission_weight = emission_weight / (emission_weight + 
                    objects[object_id].solid_angle_pdf(c_textures.eval_normal(it, material_id), ray.d, min_dist) * 
                    hit_emitter * (b > 0) * ray.non_delta());

            float direct_pdf = 1;       // direct_pdf is the product of light_sampling_pdf and emitter_pdf
            // (2) check if the ray hits an emitter
            Vec4 direct_comp = throughput * c_emitter[emitter_id]->eval_le(&ray.d, &it);
            radiance += direct_comp * emission_weight;

            Emitter* emitter = sample_emitter(sampler, direct_pdf, num_emitter, emitter_id);
            // (3) sample a point on the emitter (we avoid sampling the hit emitter)
            emitter_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(sampler.discrete1D(), direct_pdf);
            emitter_id = emitter_prims[emitter_id];               // extra mapping, introduced after BVH primitive reordering
            Ray shadow_ray(ray.advance(min_dist), Vec3(0, 0, 0));
            // use ray.o to avoid creating another shadow_int variable
            shadow_ray.d = emitter->sample(
                shadow_ray.o, it.shading_norm, direct_comp, direct_pdf, sampler.next2D(), verts, norms, uvs, emitter_id
            ) - shadow_ray.o;
            
            float emit_len_mis = shadow_ray.d.length();
            shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direction

            // (3) NEE scene intersection test (possible warp divergence, but... nevermind)
            if (emitter != c_emitter[0] &&
                occlusion_test_bvh(shadow_ray, bvh_leaves, nodes, s_cached, 
                            verts, node_num, cache_num, emit_len_mis - EPSILON)
            ) {
                // MIS for BSDF / light sampling, to achieve better rendering
                // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
                emit_len_mis = direct_pdf + c_material[material_id]->pdf(it, shadow_ray.d, ray.d, material_id) * emitter->non_delta();
                radiance += throughput * direct_comp * c_material[material_id]->eval(it, shadow_ray.d, ray.d, material_id) * \
                    (float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
                // numerical guard, in case emit_len_mis is 0
            }

            // step 4: sample a new ray direction, bounce the 
            BSDFFlag sampled_lobe = BSDFFlag::BSDF_NONE;
            ray.o = std::move(shadow_ray.o);
            ray.d = c_material[material_id]->sample_dir(ray.d, it, throughput, emission_weight, sampler, sampled_lobe, material_id);
            ray.set_delta((BSDFFlag::BSDF_SPECULAR & sampled_lobe) > 0);


            if (radiance.numeric_err())
                radiance.fill(0);
            
            // using BVH enables breaking, since there is no within-loop synchronization
            diff_b  += (BSDFFlag::BSDF_DIFFUSE  & sampled_lobe) > 0;
            spec_b  += (BSDFFlag::BSDF_SPECULAR & sampled_lobe) > 0;
            trans_b += (BSDFFlag::BSDF_TRANSMIT & sampled_lobe) > 0;
            if (diff_b  >= md_params.max_diffuse  || 
                spec_b  >= md_params.max_specular || 
                trans_b >= md_params.max_tranmit
            ) break;
            float max_value = throughput.max_elem_3d();
            if (b >= RR_BOUNCE && max_value < RR_THRESHOLD) {
                if (sampler.next1D() > max_value || max_value < THP_EPS) break;
                throughput *= 1. / max_value;
            }
        } else {
            radiance += throughput * c_emitter[envmap_id]->eval_le(&ray.d);
            break;
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
    int node_num,
    int accum_cnt,
    int cache_num,
    int envmap_index,
    bool gamma_corr
);

template CPT_KERNEL void render_pt_kernel<false>(
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
    int node_num,
    int accum_cnt,
    int cache_num,
    int envmap_index,
    bool gamma_corr
);