/**
 * Megakernel Volumetric Path Tracing (Implementation)
 * @date: 2025.2.7
 * @author: Qianyue He
*/
#include "core/max_depth.h"
#include "core/textures.cuh"
#include "core/camera_model.cuh"
#include "renderer/megakernel_vpt.cuh"

static constexpr int RR_BOUNCE = 1;
static constexpr float RR_THRESHOLD = 0.1;

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
    int num_emitter,
    int seed_offset,
    int cam_vol_idx,
    int node_num,
    int accum_cnt,
    int cache_num,
    int envmap_id,
    bool gamma_corr
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    // You see, I only provide one cam_vol_idx, meaning that camera itself can not be placed in the inner levels
    // of nested volumes. Also, this cam_vol_idx is fixed, camera movement will not update it (might be incorrect)
    BankStack nested_vols(cam_vol_idx);
    Sampler sampler(px + py * image.w(), seed_offset);
    // step 1: generate ray

    // step 2: bouncing around the scene until the max depth is reached
    int min_index = -1, diff_b = 0, spec_b = 0, trans_b = 0, volm_b = 0;
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
    
    for (int b = 0; b < md_params.max_depth; b++) {
        float prim_u = 0, prim_v = 0, min_dist = MAX_DIST;
        min_index = -1;
        // ============= step 1: ray intersection =================
        int obj_info = INVALID_OBJ;
        min_dist = ray_intersect_bvh(
            ray, bvh_leaves, nodes, s_cached, 
            verts, min_index, obj_info, 
            prim_u, prim_v, node_num, cache_num, min_dist
        );

        if (min_index <= 0) {       // definitely not inside the volume
            radiance += throughput * c_emitter[envmap_id]->eval_le(&ray.d);
            break;
        }

        bool is_triangle = true, hit_emitter = false;
        int material_id = 0, emitter_id = -1, hit_med_idx = 0, object_id = extract_tracing_info(obj_info, hit_med_idx, is_triangle);
        objects[object_id].unpack(material_id, emitter_id);
        hit_emitter = emitter_id > 0;

        const Medium* medium  = media[nested_vols.top()];        
        MediumSample md = medium->sample(ray, sampler);

        Vec4 direct_comp(0, 1), nee_tr(0, 1);

        // ====================== Sample Emitter ====================
        // (1) randomly pick one emitter
        
        float direct_pdf = 1;       // direct_pdf is the product of light_sampling_pdf and emitter_pdf
        Emitter* emitter = sample_emitter(sampler, direct_pdf, num_emitter, md.flag > 0 ? 0 : emitter_id);
        // (2) sample a point on the emitter (we avoid sampling the hit emitter)
        emitter_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(sampler.discrete1D(), direct_pdf);
        emitter_id = emitter_prims[emitter_id];               // extra mapping, introduced after BVH primitive reordering
        Ray shadow_ray(ray.advance(min_dist), Vec3(0, 0, 0));

        // sacrificing the hemispherical sampling for envmap
        shadow_ray.d = emitter->sample(
            shadow_ray.o, Vec3(0, 0, 1), direct_comp, direct_pdf, sampler.next2D(), verts, norms, uvs, emitter_id
        ) - shadow_ray.o;
        
        float emit_len_mis = shadow_ray.d.length();
        shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direction

        throughput *= md.local_thp;
        // After the emitter is sampled, we first perform NEE transmittance testing, for surface event
        // we need to test whether the ray hit the emitter. For medium event, emission grid should be tested
        // but I currently don't know how. After the transmittance is tested, we can 

        if (emitter != c_emitter[0]) {
            nee_tr = occlusion_transmittance_estimate(
                shadow_ray, sampler, bvh_leaves, nodes, 
                s_cached, verts, norms, media, node_num, 
                cache_num, nested_vols, emit_len_mis - EPSILON 
            );
        }
        // even if the emitter is c_emitter[0], we are still going to evaluate direct component, in order to
        // introduce fewer diverging threads. Evaluating direct component is not a bottleneck
        ScatterStateFlag sampled_lobe = ScatterStateFlag::BSDF_NONE;
        if (md.flag > 0) {  // medium event
            // volume measure to solid angle measure, just use the inverse squared distance
            emission_weight = emission_weight / (emission_weight + min_dist * min_dist * hit_emitter * (b > 0) * ray.non_delta());

            // TODO: Emission grid direct component testing: I don't know how it works yet
            // Maybe, just query the emission grid and get the radiance (scaled by absorption factor)
            // radiance += throughput * c_emitter[emitter_id]->eval_le(&ray.d, &it) * emission_weight;

            float phase_pdf = medium->eval(shadow_ray.d, ray.d);
            emit_len_mis = direct_pdf + phase_pdf;
            radiance += nee_tr * throughput * direct_comp * \
                (phase_pdf * float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));

            // Bounce the ray via material scattering
            sampled_lobe = ScatterStateFlag::SCAT_VOLUME;
            ray.d = medium->scatter(ray.d, throughput, sampler);

            ray.set_delta(false);               // currently, there is no delta lobe for phase function
        } else {
            auto it = Primitive::get_interaction(verts, norms, uvs, ray.advance(min_dist), prim_u, prim_v, min_index, is_triangle);

            int material_id = 0, emitter_id = -1;
            objects[object_id].unpack(material_id, emitter_id);
            hit_emitter = emitter_id > 0;

            // emitter MIS
            emission_weight = emission_weight / (emission_weight + 
                    objects[object_id].solid_angle_pdf(c_textures.eval_normal(it, material_id), ray.d, min_dist) * 
                    hit_emitter * (b > 0) * ray.non_delta());

            // check if the ray hits an emitter
            radiance += throughput * c_emitter[emitter_id]->eval_le(&ray.d, &it) * emission_weight;

            // MIS for BSDF / light sampling, to achieve better rendering
            // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
            emit_len_mis = direct_pdf + c_material[material_id]->pdf(it, shadow_ray.d, ray.d, material_id) * emitter->non_delta();
            radiance += nee_tr * throughput * direct_comp * c_material[material_id]->eval(it, shadow_ray.d, ray.d, material_id) * \
                (float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));

            // Bounce the ray via material scattering
            
            bool same_hemisphere = ray.d.dot(it.shading_norm) > 0;  // incident ray, whether the direction is in the same hemisphere with normal
            ray.d = c_material[material_id]->sample_dir(ray.d, it, throughput, emission_weight, sampler, sampled_lobe, material_id);
            ray.set_delta((ScatterStateFlag::BSDF_SPECULAR & sampled_lobe) > 0);

            // After the ray is scattered, check the scattered ray direction and normal to tell whether we are exiting or entering the medium
            // if the one of the incident | exiting ray is not in the same hemisphere with normal (and the other is), this will mean we are
            // not penetrating a medium interface, otherwise we should update the nesting_vols
            if ((it.shading_norm.dot(ray.d) > 0 ^ same_hemisphere) == false) {
                // necessary branch, sigh... hate branches
                if (same_hemisphere) {  // if we are exiting from a medium
                    nested_vols.pop();
                } else {                // if we are entering a medium
                    nested_vols.push(hit_med_idx);
                }
            }
        }
        ray.o = std::move(shadow_ray.o);
        
        if (radiance.numeric_err())
            radiance.fill(0);
        
        // using BVH enables breaking, since there is no within-loop synchronization
        diff_b  += (ScatterStateFlag::BSDF_DIFFUSE  & sampled_lobe) > 0;
        spec_b  += (ScatterStateFlag::BSDF_SPECULAR & sampled_lobe) > 0;
        trans_b += (ScatterStateFlag::BSDF_TRANSMIT & sampled_lobe) > 0;
        volm_b  += (ScatterStateFlag::SCAT_VOLUME   & sampled_lobe) > 0;
        if (diff_b  >= md_params.max_diffuse  || 
            spec_b  >= md_params.max_specular || 
            trans_b >= md_params.max_tranmit  ||
            volm_b  >= md_params.max_volume
        ) break;
        float max_value = throughput.max_elem_3d();
        if (max_value < THP_EPS) break;
        if (b >= RR_BOUNCE && max_value < RR_THRESHOLD) {
            if (sampler.next1D() > max_value) break;
            throughput *= 1. / max_value;
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

template CPT_KERNEL void render_vpt_kernel<true>(
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
    int num_emitter,
    int seed_offset,
    int cam_vol_idx,
    int node_num,
    int accum_cnt,
    int cache_num,
    int envmap_id,
    bool gamma_corr
);

template CPT_KERNEL void render_vpt_kernel<false>(
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
    int num_emitter,
    int seed_offset,
    int cam_vol_idx,
    int node_num,
    int accum_cnt,
    int cache_num,
    int envmap_id,
    bool gamma_corr
);