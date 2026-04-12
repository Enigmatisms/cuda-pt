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

/**
 * @author: Qianyue He
 * @brief Megakernel ReSTIR DI Path Tracing (Implementation)
 * @date: 2025.4.10
 *
 * Based on megakernel_pt.cu with volumetric support from megakernel_vpt.cu
 * and ReSTIR direct illumination at the first bounce.
 */
#include "core/camera_model.cuh"
#include "core/max_depth.h"
#include "core/textures.cuh"
#include "renderer/megakernel_restir.cuh"

static constexpr int RR_BOUNCE = 2;
static constexpr float RR_THRESHOLD = 0.1;

// ─── Volumetric helpers (from megakernel_vpt.cu) ────────────────────

CPT_GPU_INLINE int extract_medium_info(int obj_idx, bool &alpha_mask) {
    alpha_mask = (obj_idx & 0x40000000) == 0x40000000;
    return (obj_idx >> 20) & 0x000000ff;
}

CPT_GPU_INLINE int extract_tracing_info(int obj_idx, int &hit_med_idx,
                                        bool &is_triangle) {
    is_triangle = (obj_idx & 0x80000000) == 0;
    hit_med_idx = (obj_idx >> 20) & 0x000000ff;
    return obj_idx & 0x000fffff;
}

struct BankStack {
    uchar4 data;

    CPT_GPU BankStack() {}
    CPT_GPU BankStack(int val) : data{0, 0, 0, 0} {
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
            data.x++;
            *((&data.x) + data.x) = val;
        }
    }

    CPT_GPU_INLINE int pop() {
        int res = 0;
        if (data.x > 0) {
            res = *((&data.x) + data.x);
            data.x--;
        }
        return res;
    }
};

// Non-binary occlusion test with transmittance accumulation for volumes
inline CPT_GPU Vec4 occlusion_transmittance_estimate(
    Ray ray, Sampler &sp, const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes, ConstF4Ptr cached_nodes,
    const PrecomputedArray &verts, const NormalArray norms,
    MediumPtrArray media, const int node_num, const int cache_num,
    BankStack nested_vols, float max_dist) {
    float total_dist = 0;
    const Vec3 inv_d = ray.d.rcp();
    Vec4 Tr(1);
    while (total_dist < max_dist) {
        int node_idx = 0, min_index = -1;
        float aabb_tmin = 0, prim_u = 0, prim_v = 0,
              min_dist = max_dist - total_dist;
        int min_obj_info = INVALID_OBJ;
        Vec3 o_div = ray.o * inv_d;
        for (int i = 0; i < cache_num;) {
            const CompactNode node(cached_nodes[i]);
            bool intersect_node =
                node.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < min_dist;
            int all_offset = node.get_cached_offset(),
                gmem_index = node.get_gmem_index();
            int increment = (!intersect_node) * all_offset +
                            int(intersect_node && all_offset != 1);
            intersect_node = intersect_node && all_offset == 1;
            i = intersect_node ? cache_num : (i + increment);
            node_idx = intersect_node ? gmem_index : node_idx;
        }
        while (node_idx < node_num) {
            const LinearNode node(tex1Dfetch<float4>(nodes, 2 * node_idx),
                                  tex1Dfetch<float4>(nodes, 2 * node_idx + 1));
            int beg_idx = 0, end_idx = 0;
            node.get_range(beg_idx, end_idx);
            bool intersect_node =
                node.aabb.intersect(inv_d, o_div, aabb_tmin) &&
                aabb_tmin < min_dist;
            node_idx += (!intersect_node) * (end_idx < 0 ? -end_idx : 1) +
                        int(intersect_node);
            end_idx = intersect_node ? end_idx + beg_idx : 0;
            for (int idx = beg_idx; idx < end_idx; idx++) {
                bool is_triangle = true;
                int obj_info = tex1Dfetch<int>(bvh_leaves, idx);
                int obj_idx = extract_object_info(obj_info, is_triangle);
#ifdef TRIANGLE_ONLY
                float it_u = 0, it_v = 0,
                      dist = Primitive::intersect(ray, verts, idx, it_u, it_v);
#else
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
        bool is_alpha_mask = false;
        int active_medium = extract_medium_info(min_obj_info, is_alpha_mask);
        if (active_medium == 0 || is_alpha_mask == false) {
            Tr.fill(0);
            break;
        }
        bool is_in_medium =
            min_index >= 0 &&
            norms.eval(max(min_index, 0), prim_u, prim_v).dot(ray.d) > 0;
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

// ─── Main ReSTIR DI Kernel ──────────────────────────────────────────

template <bool render_once>
CPT_KERNEL void render_pt_restir_kernel(
    const DeviceCamera &dev_cam, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    MediumPtrArray media, ConstObjPtr objects, ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, DeviceImage image, const MaxDepthParams md_params,
    float *__restrict__ output_buffer, float *__restrict__ var_buffer,
    int num_emitter, int seed_offset, int cam_vol_idx, int node_num,
    int accum_cnt, int cache_num, int envmap_id, bool gamma_corr,
    const ReSTIRReservoir *__restrict__ prev_reservoirs,
    ReSTIRReservoir *__restrict__ curr_reservoirs,
    ReSTIRGBuffer *__restrict__ g_buffer, bool temporal_reuse) {

    int px = threadIdx.x + blockIdx.x * blockDim.x,
        py = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ BankStack nested_vol_arr[128];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    nested_vol_arr[tid] = BankStack(cam_vol_idx);

    extern __shared__ uint4 s_cached[];
    if (tid < cache_num) {
        s_cached[tid] = cached_nodes[tid];
        int offset_tid = tid + blockDim.x * blockDim.y;
        if (offset_tid < cache_num)
            s_cached[offset_tid] = cached_nodes[offset_tid];
    }
    __syncthreads();

    Sampler sampler(px + py * image.w(), seed_offset);
    int min_index = -1, diff_b = 0, spec_b = 0, trans_b = 0, volm_b = 0;
    int pix_idx = px + py * image.w();
    Ray ray = dev_cam.generate_ray(px, py, sampler);

    // Invalidate GBuffer entry — will be set valid if ReSTIR applies
    if (g_buffer)
        g_buffer[pix_idx].is_valid = 0;

    Vec4 throughput(1, 1, 1), radiance(0, 0, 0);
    float emission_weight = 1.f;

    for (int b = 0; b < md_params.max_depth; b++) {
        float prim_u = 0, prim_v = 0;
        ray.hit_t = MAX_DIST;
        min_index = -1;
        int obj_info = INVALID_OBJ;
        ray.hit_t = ray_intersect_bvh(ray, bvh_leaves, nodes, s_cached, verts,
                                      min_index, obj_info, prim_u, prim_v,
                                      node_num, cache_num, ray.hit_t);

        if (min_index < 0) {
            radiance += throughput * c_emitter[envmap_id]->eval_le(&ray.d);
            break;
        }

        bool is_triangle = true, alpha_mask = false;
        int material_id = 0, emitter_id = -1, hit_med_idx = 0,
            object_id =
                extract_tracing_info(obj_info, hit_med_idx, is_triangle);
        objects[object_id].unpack(material_id, emitter_id);

        // ─── Medium distance sampling ──────────────────────────────
        const Medium *medium = media[nested_vol_arr[tid].top()];
        MediumSample md = medium->sample(ray, sampler);
        throughput *= md.local_thp;

        // Alpha-mask culling
        extract_medium_info(obj_info, alpha_mask);
        if (alpha_mask && md.flag == 0) {
            ray.o = ray.advance(md.dist);
            auto it = Primitive::get_interaction(
                verts, norms, uvs, ray.advance(ray.hit_t), prim_u, prim_v,
                min_index, is_triangle);
            bool same_hemisphere = ray.d.dot(it.shading_norm) > 0;
            if ((it.shading_norm.dot(ray.d) > 0 ^ same_hemisphere) == false) {
                if (same_hemisphere) {
                    nested_vol_arr[tid].pop();
                } else {
                    nested_vol_arr[tid].push(hit_med_idx);
                }
            }
            continue;
        }

        // Compute hit position and interaction (needed for both NEE paths)
        Vec3 hit_pos = ray.advance(md.flag > 0 ? md.dist : ray.hit_t);
        Interaction it(Vec3(0, 0, 1), Vec2Half(0.f));
        if (md.flag == 0) {
            it = Primitive::get_interaction(verts, norms, uvs, hit_pos, prim_u,
                                            prim_v, min_index, is_triangle);
        }

        // ─── Emitter hit evaluation ───────────────────────────────
        bool hit_emitter = emitter_id > 0;
        if (md.flag == 0) {
            emission_weight =
                emission_weight /
                (emission_weight + objects[object_id].solid_angle_pdf(
                                       c_textures.eval_normal(it, material_id),
                                       ray.d, ray.hit_t) *
                                       hit_emitter * (b > 0) * ray.non_delta());
            radiance += throughput *
                        c_emitter[emitter_id]->eval_le(&ray.d, &it) *
                        emission_weight;
        }

        // ─── Direct Illumination ──────────────────────────────────
        // ReSTIR only applies at first bounce for non-specular events
        bool apply_restir = (b == 0) && (md.flag > 0 || ray.non_delta());
        ScatterStateFlag sampled_lobe = ScatterStateFlag::BSDF_NONE;

        if (apply_restir) {
            // ═══════════════════════════════════════════════════════
            // ReSTIR DI: Initial Sampling
            // ═══════════════════════════════════════════════════════
            ReSTIRReservoir reservoir = {};

            int skip_id = md.flag > 0 ? 0 : emitter_id;
            int eff_num = num_emitter - (skip_id > 0 && num_emitter > 1);
            float emit_pdf = 1.f / float(eff_num);

            for (int m = 0; m < md_params.restir_m; m++) {
                float direct_pdf = emit_pdf;
                uint32_t emit_id =
                    (sampler.discrete1D() % uint32_t(eff_num)) + 1;
                emit_id += emit_id >= skip_id && skip_id > 0;
                Emitter *emitter = c_emitter[emit_id];
                if (!emitter || emitter == c_emitter[0])
                    continue;

                int emit_obj_ref = emitter->get_obj_ref();
                int emitter_prim_id =
                    objects[emit_obj_ref].sample_emitter_primitive(
                        sampler.discrete1D(), direct_pdf);
                emitter_prim_id = emitter_prims[emitter_prim_id];
                float q_area = direct_pdf; // Capture BEFORE emitter->sample()

                Vec4 direct_comp;
                Vec3 sampled_pos = emitter->sample(
                    hit_pos, md.flag > 0 ? Vec3(0, 0, 1) : it.shading_norm,
                    direct_comp, direct_pdf, sampler.next2D(), verts, norms,
                    uvs, emitter_prim_id);

                Vec3 dir = sampled_pos - hit_pos;
                float dist = dir.length();
                dir *= __frcp_rn(max(dist, EPSILON));

                // h = direct_comp * bsdf_eval / direct_pdf
                // Reject candidates with near-zero PDF to avoid bias
                // from guard_div inflating w_sum
                Vec4 h(0);
                float w = 0.f;
                if (direct_pdf > EPSILON) {
                    Vec4 bsdf_or_phase;
                    if (md.flag > 0) {
                        float pv = medium->eval(dir, ray.d);
                        bsdf_or_phase = Vec4(pv, 1.f);
                    } else {
                        bsdf_or_phase = c_material[material_id]->eval(
                            it, dir, ray.d, material_id);
                    }
                    h = direct_comp * bsdf_or_phase * __frcp_rn(direct_pdf);
                    w = restir_luminance(h);
                }

                // emit_id is the 1-based emitter index (computed above)
                int emitter_idx = emit_id;

                // Light normal for reuse reevaluation
                // Point source detection: PointSource::sample() doesn't modify
                // direct_pdf (it takes float& but never writes to it), so if
                // direct_pdf still equals q_area after sample(), it's a point
                // source. AreaSource/SpotSource/EnvMap all modify direct_pdf.
                Vec3 light_norm(0);
                bool is_point_source =
                    (direct_pdf == q_area) && emitter_idx != envmap_id;
                if (!is_point_source && emitter_idx != envmap_id) {
                    light_norm = norms.eval(emitter_prim_id, 0.33f, 0.33f);
                }

                // emitter_type: 0=area, 1=envmap, 2=point
                int emitter_type = 0;
                if (emitter_idx == envmap_id)
                    emitter_type = 1;
                else if (is_point_source)
                    emitter_type = 2;

                // For point sources: store raw Le (un-attenuated) so
                // restir_compute_h can correctly recompute distance attenuation
                // at a new hit point.
                Vec4 stored_Le = direct_comp;
                if (is_point_source) {
                    stored_Le = emitter->get_le();
                }

                ReSTIRSample s = {emitter_type == 1 ? dir : sampled_pos,
                                  light_norm,
                                  stored_Le,
                                  q_area,
                                  emitter_idx,
                                  emitter_prim_id,
                                  emitter_type};
                restir_update_reservoir(reservoir, s, w, sampler.next1D());
            }

            // ═══════════════════════════════════════════════════════
            // Build GBuffer (shared by temporal reuse, resolve, storage)
            // ═══════════════════════════════════════════════════════
            ReSTIRGBuffer cur_gb;
            cur_gb.hit_pos = hit_pos;
            cur_gb.shading_norm = md.flag > 0 ? Vec3(0, 0, 1) : it.shading_norm;
            cur_gb.ray_d = ray.d;
            cur_gb.uv_coord = md.flag > 0 ? Vec2Half(0.f) : it.uv_coord;
            cur_gb.material_id = material_id;
            cur_gb.hit_dist = md.flag > 0 ? md.dist : ray.hit_t;
            cur_gb.vol_stack = nested_vol_arr[tid].data;
            cur_gb.is_volume = (int8_t)(md.flag > 0);
            cur_gb.is_valid = 1;

            // ═══════════════════════════════════════════════════════
            // ReSTIR DI: Temporal Reuse
            // ═══════════════════════════════════════════════════════
            if (temporal_reuse && prev_reservoirs) {
                ReSTIRReservoir prev = prev_reservoirs[pix_idx];
                if (prev.M > 0 && prev.sample.emitter_idx > 0) {
                    Vec4 h_prev =
                        restir_compute_h(prev.sample, cur_gb, media, envmap_id);
                    float w_prev = restir_luminance(h_prev);
                    // Adjusted w_sum: M_prev * p_hat_at_current_point
                    restir_merge_reservoirs(
                        reservoir, prev, float(prev.M) * w_prev,
                        sampler.next1D());
                }
            }

            // ═══════════════════════════════════════════════════════
            // ReSTIR DI: Resolve — trace single shadow ray
            // ═══════════════════════════════════════════════════════

            if (reservoir.w_sum > 0.f && reservoir.sample.emitter_idx > 0) {

                // Recompute h for the selected sample
                Vec4 h = restir_compute_h(reservoir.sample, cur_gb, media,
                                          envmap_id);
                float p_hat = restir_luminance(h);

                if (p_hat > 0.f) {
                    // Shadow ray direction and distance
                    Vec3 shadow_dir;
                    float shadow_dist;
                    if (reservoir.sample.emitter_type == 1) {
                        shadow_dir = reservoir.sample.light_pos;
                        shadow_dist = ENVMAP_DIST;
                    } else {
                        shadow_dir = reservoir.sample.light_pos - hit_pos;
                        shadow_dist = shadow_dir.length();
                        shadow_dir *= __frcp_rn(max(shadow_dist, EPSILON));
                    }

                    // Visibility test
                    float vis = 0.f;
                    Ray shadow_ray(hit_pos, shadow_dir);
                    if (md.flag > 0) {
                        Vec4 Tr = occlusion_transmittance_estimate(
                            shadow_ray, sampler, bvh_leaves, nodes, s_cached,
                            verts, norms, media, node_num, cache_num,
                            nested_vol_arr[tid], shadow_dist - EPSILON);
                        vis = restir_luminance(Tr);
                    } else {
                        vis =
                            occlusion_test_bvh(shadow_ray, bvh_leaves, nodes,
                                               s_cached, verts, node_num,
                                               cache_num, shadow_dist - EPSILON)
                                ? 1.f
                                : 0.f;
                    }

                    if (vis > 0.f) {
                        // Effective RIS PDF in solid-angle measure:
                        // p_ris_eff = M * p_hat * direct_pdf / w_sum
                        // where direct_pdf is recomputed from stored q_area
                        // and current hit point geometry.
                        float direct_pdf = 0.f;
                        if (reservoir.sample.emitter_type == 0) {
                            // Area light: direct_pdf = q_area * r^2 / cos_light
                            float dist2 = shadow_dist * shadow_dist;
                            float cos_light =
                                max(reservoir.sample.light_normal.dot(
                                        -shadow_dir),
                                    0.f);
                            direct_pdf =
                                reservoir.sample.q_area * dist2 /
                                max(cos_light, EPSILON);
                        } else if (reservoir.sample.emitter_type == 2) {
                            // Point source: direct_pdf = q_area (unmodified)
                            direct_pdf = reservoir.sample.q_area;
                        }
                        float p_ris_eff =
                            float(reservoir.M) * p_hat * direct_pdf /
                            reservoir.w_sum;
                        float bsdf_pdf_val = 0.f;
                        Emitter *sel_emitter =
                            c_emitter[reservoir.sample.emitter_idx];
                        if (reservoir.sample.emitter_type != 1 &&
                            sel_emitter->non_delta()) {
                            bsdf_pdf_val = c_material[material_id]->pdf(
                                it, shadow_dir, ray.d, material_id);
                        }
                        float mis_denom = p_ris_eff + bsdf_pdf_val;
                        float mis_weight =
                            (mis_denom > EPSILON) ? p_ris_eff / mis_denom
                                                  : 1.f;

                        // L = throughput * (w_sum/M) * vis * (h/p_hat) * mis
                        float rcp_p_hat = __frcp_rn(p_hat);
                        radiance += throughput *
                                    (reservoir.w_sum / float(reservoir.M)) *
                                    vis * h * (mis_weight * rcp_p_hat);
                    }
                }
            }

            // Store GBuffer and reservoir for future reuse (always)
            if (g_buffer)
                g_buffer[pix_idx] = cur_gb;
            if (curr_reservoirs)
                curr_reservoirs[pix_idx] = reservoir;

            // ═══════════════════════════════════════════════════════
            // BSDF / Phase function sampling (continues the path)
            // ═══════════════════════════════════════════════════════
            if (md.flag > 0) {
                sampled_lobe = ScatterStateFlag::SCAT_VOLUME;
                ray.d = medium->scatter(ray.d, throughput, sampler,
                                        emission_weight);
                ray.set_delta(false);
            } else {
                bool same_hemisphere = ray.d.dot(it.shading_norm) > 0;
                ray.d = c_material[material_id]->sample_dir(
                    ray.d, it, throughput, emission_weight, sampler,
                    sampled_lobe, material_id);
                ray.set_delta((ScatterStateFlag::BSDF_SPECULAR & sampled_lobe) >
                              0);

                // Volume stack update
                if ((it.shading_norm.dot(ray.d) > 0 ^ same_hemisphere) ==
                    false) {
                    if (same_hemisphere) {
                        nested_vol_arr[tid].pop();
                    } else {
                        nested_vol_arr[tid].push(hit_med_idx);
                    }
                }
            }
            ray.o = hit_pos;
        } else {
            // ═══════════════════════════════════════════════════════
            // Classic NEE (b > 0 or specular first bounce)
            // Same structure as megakernel_vpt.cu
            // ═══════════════════════════════════════════════════════
            Vec4 direct_comp(0, 1), nee_tr(0, 1);
            float direct_pdf = 1;
            Emitter *emitter = sample_emitter(sampler, direct_pdf, num_emitter,
                                              md.flag > 0 ? 0 : emitter_id);

            int emitter_prim_id =
                objects[emitter->get_obj_ref()].sample_emitter_primitive(
                    sampler.discrete1D(), direct_pdf);
            emitter_prim_id = emitter_prims[emitter_prim_id];

            Ray shadow_ray(hit_pos, Vec3(0, 0, 0));
            shadow_ray.d =
                emitter->sample(shadow_ray.o,
                                md.flag > 0 ? Vec3(0, 0, 1) : it.shading_norm,
                                direct_comp, direct_pdf, sampler.next2D(),
                                verts, norms, uvs, emitter_prim_id) -
                shadow_ray.o;

            float emit_len_mis = shadow_ray.d.length();
            shadow_ray.d *= __frcp_rn(emit_len_mis);

            // Visibility test
            if (emitter != c_emitter[0]) {
                if (md.flag > 0) {
                    nee_tr = occlusion_transmittance_estimate(
                        shadow_ray, sampler, bvh_leaves, nodes, s_cached, verts,
                        norms, media, node_num, cache_num, nested_vol_arr[tid],
                        emit_len_mis - EPSILON);
                } else {
                    nee_tr = occlusion_test_bvh(
                                 shadow_ray, bvh_leaves, nodes, s_cached, verts,
                                 node_num, cache_num, emit_len_mis - EPSILON)
                                 ? Vec4(1)
                                 : Vec4(0);
                }
            }

            Vec4 nee_thp;
            if (md.flag > 0) {
                // Volume event
                float phase_pdf = medium->eval(shadow_ray.d, ray.d);
                nee_thp = Vec4(phase_pdf, 1.f);
                emit_len_mis = direct_pdf + phase_pdf;

                sampled_lobe = ScatterStateFlag::SCAT_VOLUME;
                ray.d = medium->scatter(ray.d, throughput, sampler,
                                        emission_weight);
                ray.set_delta(false);
            } else {
                // Surface event
                emit_len_mis =
                    direct_pdf + c_material[material_id]->pdf(
                                     it, shadow_ray.d, ray.d, material_id) *
                                     emitter->non_delta();
                nee_thp = c_material[material_id]->eval(it, shadow_ray.d, ray.d,
                                                        material_id);

                bool same_hemisphere = ray.d.dot(it.shading_norm) > 0;
                ray.d = c_material[material_id]->sample_dir(
                    ray.d, it, throughput, emission_weight, sampler,
                    sampled_lobe, material_id);
                ray.set_delta((ScatterStateFlag::BSDF_SPECULAR & sampled_lobe) >
                              0);

                // Volume stack update
                if ((it.shading_norm.dot(ray.d) > 0 ^ same_hemisphere) ==
                    false) {
                    if (same_hemisphere) {
                        nested_vol_arr[tid].pop();
                    } else {
                        nested_vol_arr[tid].push(hit_med_idx);
                    }
                }
            }

            radiance +=
                nee_tr * throughput * direct_comp *
                (nee_thp * float(emit_len_mis > EPSILON) *
                 __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
            ray.o = hit_pos;
        }

        if (radiance.numeric_err())
            radiance.fill(0);

        diff_b += (ScatterStateFlag::BSDF_DIFFUSE & sampled_lobe) > 0;
        spec_b += (ScatterStateFlag::BSDF_SPECULAR & sampled_lobe) > 0;
        trans_b += (ScatterStateFlag::BSDF_TRANSMIT & sampled_lobe) > 0;
        volm_b += (ScatterStateFlag::SCAT_VOLUME & sampled_lobe) > 0;
        if (diff_b >= md_params.max_diffuse ||
            spec_b >= md_params.max_specular ||
            trans_b >= md_params.max_tranmit || volm_b >= md_params.max_volume)
            break;
        float max_value = throughput.max_elem_3d();
        if (max_value < THP_EPS)
            break;
        if (b >= RR_BOUNCE && max_value < RR_THRESHOLD) {
            if (sampler.next1D() > max_value)
                break;
            throughput *= 1. / max_value;
        }
    }

    if constexpr (render_once) {
        auto local_v = image(px, py);
        if (var_buffer)
            estimate_variance(var_buffer, local_v, radiance, px, py, image.w(),
                              accum_cnt);
        local_v += radiance;
        image(px, py) = local_v;
        local_v *= 1.f / float(accum_cnt);
        local_v = gamma_corr ? local_v.gamma_corr() : local_v;
        FLOAT4(output_buffer[(px + py * image.w()) << 2]) = float4(local_v);
    } else {
        image(px, py) += radiance;
    }
}

// ─── Explicit template instantiation ────────────────────────────────

template CPT_KERNEL void render_pt_restir_kernel<true>(
    const DeviceCamera &dev_cam, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    MediumPtrArray media, ConstObjPtr objects, ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, DeviceImage image, const MaxDepthParams md_params,
    float *__restrict__ output_buffer, float *__restrict__ var_buffer,
    int num_emitter, int seed_offset, int cam_vol_idx, int node_num,
    int accum_cnt, int cache_num, int envmap_id, bool gamma_corr,
    const ReSTIRReservoir *__restrict__ prev_reservoirs,
    ReSTIRReservoir *__restrict__ curr_reservoirs,
    ReSTIRGBuffer *__restrict__ g_buffer, bool temporal_reuse);

template CPT_KERNEL void render_pt_restir_kernel<false>(
    const DeviceCamera &dev_cam, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    MediumPtrArray media, ConstObjPtr objects, ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, DeviceImage image, const MaxDepthParams md_params,
    float *__restrict__ output_buffer, float *__restrict__ var_buffer,
    int num_emitter, int seed_offset, int cam_vol_idx, int node_num,
    int accum_cnt, int cache_num, int envmap_id, bool gamma_corr,
    const ReSTIRReservoir *__restrict__ prev_reservoirs,
    ReSTIRReservoir *__restrict__ curr_reservoirs,
    ReSTIRGBuffer *__restrict__ g_buffer, bool temporal_reuse);
