/**
 * Megakernel Path Tracing (Implementation)
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#include "core/textures.cuh"
#include "renderer/megakernel_pt.cuh"

static constexpr int RR_BOUNCE = 2;
static constexpr float RR_THRESHOLD = 0.1;

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
    int min_index = -1, diff_b = 0, spec_b = 0, trans_b = 0;
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
        uint32_t min_object_info = INVALID_OBJ;
        min_index = -1;
        // ============= step 1: ray intersection =================
        min_dist = ray_intersect_bvh(
            ray, bvh_leaves, nodes, s_cached, 
            verts, min_index, min_object_info, 
            prim_u, prim_v, node_num, cache_num, min_dist
        );

        bool is_triangle = true;
        int object_id = extract_object_info(min_object_info, is_triangle);

        // ============= step 2: local shading for indirect bounces ================
        if (min_index >= 0) {
            auto it = Primitive::get_interaction(verts, norms, uvs, ray.advance(min_dist), prim_u, prim_v, min_index, is_triangle);

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

            ScatterStateFlag sampled_lobe = ScatterStateFlag::BSDF_NONE;
            ray.o = std::move(shadow_ray.o);
            ray.d = c_material[material_id]->sample_dir(ray.d, it, throughput, emission_weight, sampler, sampled_lobe, material_id);
            ray.set_delta((ScatterStateFlag::BSDF_SPECULAR & sampled_lobe) > 0);

            if (radiance.numeric_err())
                radiance.fill(0);
            
            // using BVH enables breaking, since there is no within-loop synchronization
            diff_b  += (ScatterStateFlag::BSDF_DIFFUSE  & sampled_lobe) > 0;
            spec_b  += (ScatterStateFlag::BSDF_SPECULAR & sampled_lobe) > 0;
            trans_b += (ScatterStateFlag::BSDF_TRANSMIT & sampled_lobe) > 0;
            if (diff_b  >= md_params.max_diffuse  || 
                spec_b  >= md_params.max_specular || 
                trans_b >= md_params.max_tranmit
            ) break;
            float max_value = throughput.max_elem_3d();
            if (max_value < THP_EPS) break;
            if (b >= RR_BOUNCE && max_value < RR_THRESHOLD) {
                if (sampler.next1D() > max_value) break;
                throughput *= 1. / max_value;
            }
        } else {
            radiance += throughput * c_emitter[envmap_id]->eval_le(&ray.d);
            break;
        }
    }
    if constexpr (render_once) {
        // image will be the output buffer, there will be double buffering
        auto local_v = image(px, py);
        if (var_buffer)
            estimate_variance(var_buffer, local_v, radiance, px, py, image.w(), accum_cnt);
        local_v += radiance;
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
    float* __restrict__ var_buffer,
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
    float* __restrict__ var_buffer,
    int num_emitter,
    int seed_offset,
    int node_num,
    int accum_cnt,
    int cache_num,
    int envmap_index,
    bool gamma_corr
);