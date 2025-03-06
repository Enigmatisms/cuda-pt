/**
 * Megakernel Volumetric Path Tracing (Implementation)
 * @date: 2025.2.7
 * @author: Qianyue He
*/
#include "core/max_depth.h"
#include "core/textures.cuh"
#include "core/camera_model.cuh"
#include "renderer/megakernel_vpt.cuh"

static constexpr int RR_BOUNCE = 2;
static constexpr float RR_THRESHOLD = 0.1;

// returns medium_index and whether the object is alpha masked
CPT_GPU_INLINE int extract_medium_info(int obj_idx, bool& alpha_mask) {
    alpha_mask  = (obj_idx & 0x40000000) == 0x40000000;              // bit 31
     // extract higher 12 bits and mask the resulting lower 8bits
    return (obj_idx >> 20) & 0x000000ff;
}

CPT_GPU_INLINE int extract_tracing_info(int obj_idx, int& hit_med_idx, bool& is_triangle) {
    is_triangle = (obj_idx & 0x80000000) == 0;
     // extract higher 12 bits and mask the resulting lower 8bits
    hit_med_idx = (obj_idx >> 20) & 0x000000ff;
    return obj_idx & 0x000fffff;                            // extract low 20bits, return the object index
}

#ifdef SUPPORTS_TOF_RENDERING
CPT_GPU_INLINE bool time_in_range(const MaxDepthParams& mdp, float t) {
    return mdp.max_time <= 0 || (t < mdp.max_time && t > mdp.min_time);
}
#else
constexpr CPT_GPU_INLINE bool time_in_range(const MaxDepthParams& mdp, float t) {
    return true;
}
#endif // SUPPORTS_TOF_RENDERING

/**
 * @brief Stack with only one bank (4B), used for handling nested volumes
 * x is the ptr, if x == 0, it means that active volume is 0 (not within a volume)
 */
struct BankStack {
    uchar4 data;

    CPT_GPU BankStack() {}
    CPT_GPU BankStack(int val): data{0, 0, 0, 0} {
        if (val > 0) {
            data.x = 1;
            data.y = uint8_t(val);
        }
    }

    CPT_GPU_INLINE int top() const {
        return data.x > 0 ? *((&data.x) + data.x) : 0;
    }

    CPT_GPU_INLINE void push(uint8_t val) {
        if (data.x < 3 && val != 0xff) {
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
    const Vec3 inv_d = ray.d.rcp();
    Vec4 Tr(1);
    while (total_dist < max_dist) {
        int node_idx     = 0, min_index = -1;
        float aabb_tmin  = 0, prim_u = 0, prim_v = 0, min_dist = max_dist - total_dist;     // FIXME: precision problem
        int min_obj_info = INVALID_OBJ;
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
                int obj_info = tex1Dfetch<int>(bvh_leaves, idx);
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
            int old_active_medium = nested_vols.top();
            nested_vols.push(active_medium);
            active_medium = old_active_medium;
        }
        Tr *= media[active_medium]->transmittance(ray, sp, min_dist);
        
        total_dist += min_dist;
        ray.o = ray.advance(min_dist);
    }
    return Tr;
}

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
    __shared__ BankStack nested_vol_arr[128];

    Sampler sampler(px + py * image.w(), seed_offset);
    // step 1: generate ray

    // step 2: bouncing around the scene until the max depth is reached
    int min_index = -1, diff_b = 0, spec_b = 0, trans_b = 0, volm_b = 0;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    nested_vol_arr[tid] = BankStack(cam_vol_idx);
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
    float emission_weight = 1.f, total_dist = 0;
    
    for (int b = 0; b < md_params.max_depth; b++) {
        float prim_u = 0, prim_v = 0;
        ray.hit_t = MAX_DIST;
        min_index = -1;
        // ============= step 1: ray intersection =================
        int obj_info = INVALID_OBJ;
        ray.hit_t = ray_intersect_bvh(
            ray, bvh_leaves, nodes, s_cached, 
            verts, min_index, obj_info, 
            prim_u, prim_v, node_num, cache_num, ray.hit_t
        );

        if (min_index < 0) {       // definitely not inside the volume
            radiance += throughput * c_emitter[envmap_id]->eval_le(&ray.d);
            break;
        }

        bool is_triangle = true, hit_emitter = false, alpha_mask = false;
        int material_id = 0, emitter_id = -1, hit_med_idx = 0, object_id = extract_tracing_info(obj_info, hit_med_idx, is_triangle);
        objects[object_id].unpack(material_id, emitter_id);
        hit_emitter = emitter_id > 0;

        const Medium* medium  = media[nested_vol_arr[tid].top()];        
        MediumSample md = medium->sample(ray, sampler);
        throughput *= md.local_thp;

        extract_medium_info(obj_info, alpha_mask);
        if (alpha_mask && md.flag == 0) {               
            // surface event, forward BSDF (cullable) should not have NEE
            ray.o = ray.advance(md.dist);
            auto it = Primitive::get_interaction(verts, norms, uvs, ray.advance(ray.hit_t), prim_u, prim_v, min_index, is_triangle);
            bool same_hemisphere = ray.d.dot(it.shading_norm) > 0;
            if ((it.shading_norm.dot(ray.d) > 0 ^ same_hemisphere) == false) {
                if (same_hemisphere) {  // if we are exiting from a medium
                    nested_vol_arr[tid].pop();
                } else {                // if we are entering a medium
                    nested_vol_arr[tid].push(hit_med_idx);
                }
            }
            continue;
        }

        Vec4 direct_comp(0, 1), nee_tr(0, 1);

        // ====================== Sample Emitter ====================
        // (1) randomly pick one emitter
        
        float direct_pdf = 1;       // direct_pdf is the product of light_sampling_pdf and emitter_pdf
        Emitter* emitter = sample_emitter(sampler, direct_pdf, num_emitter, md.flag > 0 ? 0 : emitter_id);
        // (2) sample a point on the emitter (we avoid sampling the hit emitter)
        int emitter_prim_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(sampler.discrete1D(), direct_pdf);
        emitter_prim_id = emitter_prims[emitter_prim_id];               // extra mapping, introduced after BVH primitive reordering
        Ray shadow_ray(ray.advance(md.dist), Vec3(0, 0, 0));

        // sacrificing the hemispherical sampling for envmap
        shadow_ray.d = emitter->sample(
            shadow_ray.o, Vec3(0, 0, 1), direct_comp, direct_pdf, sampler.next2D(), verts, norms, uvs, emitter_prim_id
        ) - shadow_ray.o;
        
        float emit_len_mis = shadow_ray.d.length(), shadow_length = emit_len_mis;
        shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direction

        // After the emitter is sampled, we first perform NEE transmittance testing, for surface event
        // we need to test whether the ray hit the emitter. For medium event, emission grid should be tested
        // but I currently don't know how. After the transmittance is tested, we can 

        if (emitter != c_emitter[0]) {
            nee_tr = occlusion_transmittance_estimate(
                shadow_ray, sampler, bvh_leaves, nodes, 
                s_cached, verts, norms, media, node_num, 
                cache_num, nested_vol_arr[tid], emit_len_mis - EPSILON 
            );
        }
        // even if the emitter is c_emitter[0], we are still going to evaluate direct component, in order to
        // introduce fewer diverging threads. Evaluating direct component is not a bottleneck
        ScatterStateFlag sampled_lobe = ScatterStateFlag::BSDF_NONE;
        Vec4 nee_thp;
        if (md.flag > 0) {  // medium event
            CONDITION_BLOCK(time_in_range(md_params, total_dist)) {
                radiance += throughput * medium->query_emission(shadow_ray.o, sampler);
            }

            float phase_pdf = medium->eval(shadow_ray.d, ray.d);
            nee_thp = Vec4(phase_pdf, 1.f);
            emit_len_mis = direct_pdf + phase_pdf;
            // printf("Here: %d, phase_pdf: %f, (%f, %f, %f)\n", nested_vols.top(), phase_pdf, throughput.x(), nee_tr.y(), );

            // Bounce the ray via material scattering
            sampled_lobe = ScatterStateFlag::SCAT_VOLUME;
            ray.d = medium->scatter(ray.d, throughput, sampler, emission_weight);

            ray.set_delta(false);               // currently, there is no delta lobe for phase function
        } else {
            auto it = Primitive::get_interaction(verts, norms, uvs, ray.advance(ray.hit_t), prim_u, prim_v, min_index, is_triangle);
            hit_emitter = emitter_id > 0;

            // emitter MIS
            emission_weight = emission_weight / (emission_weight + 
                    objects[object_id].solid_angle_pdf(c_textures.eval_normal(it, material_id), ray.d, ray.hit_t) * 
                    hit_emitter * (b > 0) * ray.non_delta());

            // check if the ray hits an emitter
            CONDITION_BLOCK(time_in_range(md_params, total_dist + shadow_length)) {
                radiance += throughput * c_emitter[emitter_id]->eval_le(&ray.d, &it) * emission_weight;
            }

            // MIS for BSDF / light sampling, to achieve better rendering
            // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
            emit_len_mis = direct_pdf + c_material[material_id]->pdf(it, shadow_ray.d, ray.d, material_id) * emitter->non_delta();
            nee_thp = c_material[material_id]->eval(it, shadow_ray.d, ray.d, material_id);

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
                    nested_vol_arr[tid].pop();
                } else {                // if we are entering a medium
                    nested_vol_arr[tid].push(hit_med_idx);
                }
            }
        }
        // VPT time constraint
        CONDITION_BLOCK(time_in_range(md_params, total_dist + shadow_length)) {
            radiance += nee_tr * throughput * direct_comp * \
                    (nee_thp * float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
        }
        total_dist += sampled_lobe != ScatterStateFlag::BSDF_NONE ? md.dist : 0;
        CONDITION_BLOCK(md_params.max_time > 0 && total_dist >= md_params.max_time) {
            break;
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