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
 * @brief Wavefront Path Tracing (Implementation)
 * @date: 9.15.2024
 */
#include "renderer/wavefront_pt.cuh"

static constexpr int SEED_SCALER = 11467; //-4!
static constexpr int RR_BOUNCE = 2;
static constexpr float RR_THRESHOLD = 0.1;

static CPT_GPU_INLINE uint8_t compose_ray_stat(int obj_idx, bool is_active,
                                               bool is_media = false) {
    // store object index (0-62), media flag and whether it is invalid
    // the object index stored here does not consider whether the object is
    // sphere or mesh based
    obj_idx = obj_idx >= 0
                  ? obj_idx
                  : -obj_idx - 1; // tackle the sphere object index (negative)
    return (is_active ? obj_idx % 64 : 0x3f) + (is_media << 6) +
           (is_active ? 0 : 0x80);
}

/**
 * index in the index buffer is reused:
 * higher 8bits: ray_stat for sorting
 * bit0-bit5 (24-29th): material index
 * bit6 (30th): is medium event
 * bit7 (31th): is invalid (non-active), so non-active rays will be sorted to
 * higher addresses
 *
 * lower 24 bits: index buffer (our image won't be bigger than 4800 * 3200)
 */

static CPT_GPU_INLINE uint32_t get_index(const IndexBuffer idxs, int gmem_addr,
                                         uint8_t &ray_stat) {
    uint32_t index = idxs[gmem_addr];
    ray_stat = static_cast<uint8_t>(index >> 24);
    return index & 0x00ffffff;
}

static CPT_GPU_INLINE void set_index(IndexBuffer idxs, int gmem_addr, int gidx,
                                     uint8_t stat = 0x00) {
    idxs[gmem_addr] = (0x00ffffff & gidx) + (uint32_t(stat) << 24);
}

static CPT_GPU_INLINE void set_status(IndexBuffer idxs, int gmem_addr,
                                      uint8_t stat) {
    uint32_t index = idxs[gmem_addr];
    index &= 0x00ffffff; // clear
    idxs[gmem_addr] = index | (uint32_t(stat) << 24);
}

CPT_KERNEL void raygen_primary_hit_shader(
    const DeviceCamera &dev_cam, PayLoadBufferSoA payloads,
    const PrecomputedArray verts, const NormalArray norms,
    const ConstBuffer<PackedHalf2> uvs, ConstObjPtr objects,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, IndexBuffer idx_buffer, int width, int node_num,
    int cache_num, int accum_cnt, int seed_offset, int envmap_id) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x,
              py = threadIdx.y + blockIdx.y * blockDim.y;
    const int gidx = px + py * width;

    Sampler sg(gidx, seed_offset + accum_cnt * SEED_SCALER);
    Ray ray = dev_cam.generate_ray(px, py, sg);
    ray.set_active(false);

    Interaction it; // To local register

    int min_index = -1, min_object_info = 0;
    ray.hit_t = MAX_DIST;
    float prim_u = 0, prim_v = 0;

    payloads.thp(gidx) = Vec4(1, 1, 1, 1);

    // cache near root level BVH nodes for faster traversal
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    extern __shared__ uint4 s_cached[];
    if (tid < cache_num) { // no more than 256 nodes will be cached
        s_cached[tid] = cached_nodes[tid];
        int offset_tid = tid + blockDim.x * blockDim.y;
        if (offset_tid < cache_num)
            s_cached[offset_tid] = cached_nodes[offset_tid];
    }
    __syncthreads();
    payloads.L(gidx) = Vec4(0, 1);
    payloads.pdf(gidx) = 1.f;
    payloads.set_sampler(gidx, sg);
    ray.hit_t = ray_intersect_bvh(ray, bvh_leaves, nodes, s_cached, verts,
                                  min_index, min_object_info, prim_u, prim_v,
                                  node_num, cache_num, MAX_DIST);
    bool is_triangle = true;
    int object_id = extract_object_info(min_object_info, is_triangle);

    // ============= step 2: local shading for indirect bounces ================
    if (min_index >= 0) {
        // if the ray hits nothing, or the path throughput is 0, then the ray
        // will be inactive inactive rays will only be processed in the
        // miss_shader
        ray.set_hit();
        ray.set_active(true);
        ray.set_hit_index(min_index);
        it = Primitive::get_interaction(verts, norms, uvs,
                                        ray.advance(ray.hit_t), prim_u, prim_v,
                                        min_index, is_triangle);
    } else {
        payloads.L(gidx) = c_emitter[envmap_id]->eval_le(&ray.d);
    }

    set_index(idx_buffer, gidx, gidx,
              compose_ray_stat(object_id, min_index >= 0));

    payloads.set_ray(gidx, ray);
    payloads.interaction(gidx) = it;
}

/**
 * @brief Fused shader: closest hit and miss shader
 * Except from raygen shader, all other shaders have very different shapes:
 * For example: <gridDim 1D, blockDim 1D>
 * gridDim: num_ray_payload / blockDim, blockDim = 128
 */
CPT_KERNEL void fused_closesthit_shader(
    PayLoadBufferSoA payloads, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, IndexBuffer idx_buffer, int node_num,
    int cache_num, int bounce, int envmap_id) {
    uint8_t ray_stat = 0;
    const uint32_t gmem_addr = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t gidx = get_index(idx_buffer, gmem_addr, ray_stat);

    // cache near root level BVH nodes for faster traversal
    extern __shared__ uint4 s_cached[];
    if (threadIdx.x < cache_num) { // no more than 256 nodes will be cached
        s_cached[threadIdx.x] = cached_nodes[threadIdx.x];
        int offset_tid = threadIdx.x + blockDim.x;
        if (offset_tid < cache_num)
            s_cached[offset_tid] = cached_nodes[offset_tid];
    }
    __syncthreads();

    if (ray_stat < 0x80) { // highest bit is not 1
        Ray ray = payloads.get_ray(gidx);
        ray.reset();

        float prim_u = 0, prim_v = 0;
        int min_index = -1, min_object_info = 0;
        ray.hit_t = MAX_DIST;
        ray.hit_t = ray_intersect_bvh(ray, bvh_leaves, nodes, s_cached, verts,
                                      min_index, min_object_info, prim_u,
                                      prim_v, node_num, cache_num, MAX_DIST);
        bool is_triangle = true;
        int object_id = extract_object_info(min_object_info, is_triangle);

        ray.set_hit(min_index >= 0);
        ray.set_hit_index(min_index);
        // always validates the ray
        ray.set_active(true);

        Vec4 thp = payloads.thp(gidx);
        if (min_index >= 0) {
            // if hit, first upload the interaction regardless of RR
            payloads.interaction(gidx) = Primitive::get_interaction(
                verts, norms, uvs, ray.advance(ray.hit_t), prim_u, prim_v,
                min_index, is_triangle);

            Sampler sampler = payloads.get_sampler(gidx);
            float max_value = thp.max_elem_3d();

            // Meets RR termination condition? Randomly sample termination
            if (bounce >= RR_BOUNCE && max_value < RR_THRESHOLD) {
                bool no_rr_terminate =
                    sampler.next1D() < max_value && max_value > THP_EPS;
                ray.set_active(no_rr_terminate);
                thp *= no_rr_terminate ? 1.f / max_value : 0.f;
            }
            payloads.thp(gidx) = thp;
            payloads.set_sampler(gidx, sampler);
        } else {
            payloads.L(gidx) += thp * c_emitter[envmap_id]->eval_le(&ray.d);
            ray.set_active(false);
        }
        ray_stat = compose_ray_stat(object_id, ray.is_active());
        set_status(idx_buffer, gmem_addr, ray_stat);
        // update the ray
        payloads.set_ray(gidx, ray);
    }
}

/***
 * Fusing NEE / Ray Scattering
 *
 */
CPT_KERNEL void fused_ray_bounce_shader(
    PayLoadBufferSoA payloads, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects, ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, const IndexBuffer idx_buffer, int num_emitter,
    int node_num, int cache_num, bool secondary_bounce) {
    uint8_t ray_stat = 0;
    uint32_t gmem_addr = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t gidx = get_index(idx_buffer, gmem_addr, ray_stat);

    // cache near root level BVH nodes for faster traversal
    extern __shared__ uint4 s_cached[];
    if (threadIdx.x < cache_num) { // no more than 256 nodes will be cached
        s_cached[threadIdx.x] = cached_nodes[threadIdx.x];
        int offset_tid = threadIdx.x + blockDim.x;
        if (offset_tid < cache_num)
            s_cached[offset_tid] = cached_nodes[offset_tid];
    }
    __syncthreads();

    if (ray_stat < 0x80) {
        // TODO: we need to interleave computation and memory accessing
        // Though the compiler will help us achieve this goal, but...
        // If we do this explicitly, the compiler might help us more
        Vec4 thp = payloads.thp(gidx), rdc = payloads.L(gidx);
        Ray ray = payloads.get_ray(gidx);
        Sampler sg = payloads.get_sampler(gidx);
        const Interaction it = payloads.interaction(gidx);

        int object_id = tex1Dfetch<int>(bvh_leaves, ray.hit_id());
        object_id = object_id >= 0
                        ? object_id
                        : -object_id - 1; // sphere object ID is -id - 1
        int material_id = 0, emitter_id = 0;
        objects[object_id].unpack(material_id, emitter_id);

        float direct_pdf = 1;

        Emitter *emitter =
            sample_emitter(sg, direct_pdf, num_emitter, emitter_id);
        int emit_prim_id =
            objects[emitter->get_obj_ref()].sample_emitter_primitive(
                sg.discrete1D(), direct_pdf);
        emit_prim_id =
            emitter_prims[emit_prim_id]; // extra mapping, introduced after BVH
                                         // primitive reordering
        Ray shadow_ray(ray.advance(ray.hit_t), Vec3(0, 0, 0));
        ray.o = shadow_ray.o;

        Vec4 direct_comp(0, 0, 0, 1);
        shadow_ray.d = emitter->sample(shadow_ray.o, it.shading_norm,
                                       direct_comp, direct_pdf, sg.next2D(),
                                       verts, norms, uvs, emit_prim_id) -
                       shadow_ray.o;

        float emit_len_mis = shadow_ray.d.length();
        shadow_ray.d *= __frcp_rn(emit_len_mis); // normalized direct
        // (3) NEE scene intersection test (possible warp divergence, but...
        // nevermind)
        if (emitter != c_emitter[0] &&
            occlusion_test_bvh(shadow_ray, bvh_leaves, nodes, s_cached, verts,
                               node_num, cache_num, emit_len_mis - EPSILON)) {
            // MIS for BSDF / light sampling, to achieve better rendering
            // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf +
            // material_pdf), divided by direct_pdf
            emit_len_mis =
                direct_pdf + c_material[material_id]->pdf(it, shadow_ray.d,
                                                          ray.d, material_id) *
                                 emitter->non_delta();
            rdc += thp * direct_comp *
                   c_material[material_id]->eval(it, shadow_ray.d, ray.d,
                                                 material_id) *
                   (float(emit_len_mis > EPSILON) *
                    __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
            // numerical guard, in case emit_len_mis is 0
        }

        // emitter MIS
        float pdf = payloads.pdf(gidx),
              emission_weight =
                  pdf / (pdf + objects[object_id].solid_angle_pdf(
                                   c_textures.eval_normal(it, material_id),
                                   ray.d, ray.hit_t) *
                                   (emitter_id > 0) * secondary_bounce *
                                   ray.non_delta());
        // (2) account for emission, and accumulate to payload buffer
        payloads.L(gidx) = (thp * c_emitter[emitter_id]->eval_le(&ray.d, &it)) *
                               emission_weight +
                           rdc;

        ScatterStateFlag sampled_lobe = ScatterStateFlag::BSDF_NONE;
        ray.d = c_material[material_id]->sample_dir(ray.d, it, thp, pdf, sg,
                                                    sampled_lobe, material_id);
        ray.set_delta((sampled_lobe & ScatterStateFlag::BSDF_SPECULAR) > 0);

        payloads.thp(gidx) = thp;
        payloads.set_sampler(gidx, sg);
        payloads.set_ray(gidx, ray);
        payloads.pdf(gidx) = pdf;
    }
}

/**
 * @brief This shader is used in path guiding enabled WFPT
 * This kernel handles NEE and the ray hitting an emitter
 *
 * TODO: in this kernel, we must fill in the query
 * so that the neural network can eval the NASG params for us
 * before we execute the next kernel (`guided_ray_scatter_shader`)
 */
CPT_KERNEL void
nee_direct_shader(PayLoadBufferSoA payloads, const PrecomputedArray verts,
                  const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
                  ConstObjPtr objects, ConstIndexPtr emitter_prims,
                  const cudaTextureObject_t bvh_leaves,
                  const cudaTextureObject_t nodes, ConstF4Ptr cached_nodes,
                  const IndexBuffer idx_buffer, int num_emitter, int node_num,
                  int cache_num, bool secondary_bounce
                  /**TODO: output (evaluated NASG params) */
) {
    uint8_t ray_stat = 0;
    uint32_t gmem_addr = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t gidx = get_index(idx_buffer, gmem_addr, ray_stat);

    // cache near root level BVH nodes for faster traversal
    extern __shared__ uint4 s_cached[];
    if (threadIdx.x < cache_num) { // no more than 256 nodes will be cached
        s_cached[threadIdx.x] = cached_nodes[threadIdx.x];
        int offset_tid = threadIdx.x + blockDim.x;
        if (offset_tid < cache_num)
            s_cached[offset_tid] = cached_nodes[offset_tid];
    }
    __syncthreads();

    if (ray_stat < 0x80) {
        // TODO: we need to interleave computation and memory accessing
        // Though the compiler will help us achieve this goal, but...
        // If we do this explicitly, the compiler might help us more
        Vec4 thp = payloads.thp(gidx), rdc = payloads.L(gidx);
        Ray ray = payloads.get_ray(gidx);
        Sampler sg = payloads.get_sampler(gidx);
        const Interaction it = payloads.interaction(gidx);

        int object_id = tex1Dfetch<int>(bvh_leaves, ray.hit_id());
        object_id = object_id >= 0
                        ? object_id
                        : -object_id - 1; // sphere object ID is -id - 1
        int material_id = 0, emitter_id = 0;
        objects[object_id].unpack(material_id, emitter_id);

        float direct_pdf = 1;

        Emitter *emitter =
            sample_emitter(sg, direct_pdf, num_emitter, emitter_id);
        int emit_prim_id =
            objects[emitter->get_obj_ref()].sample_emitter_primitive(
                sg.discrete1D(), direct_pdf);
        emit_prim_id =
            emitter_prims[emit_prim_id]; // extra mapping, introduced after BVH
                                         // primitive reordering
        Ray shadow_ray(ray.advance(ray.hit_t), Vec3(0, 0, 0));
        ray.o = shadow_ray.o;

        Vec4 direct_comp(0, 0, 0, 1);
        shadow_ray.d = emitter->sample(shadow_ray.o, it.shading_norm,
                                       direct_comp, direct_pdf, sg.next2D(),
                                       verts, norms, uvs, emit_prim_id) -
                       shadow_ray.o;

        float emit_len_mis = shadow_ray.d.length();
        shadow_ray.d *= __frcp_rn(emit_len_mis); // normalized direct
        // (3) NEE scene intersection test (possible warp divergence, but...
        // nevermind)
        if (emitter != c_emitter[0] &&
            occlusion_test_bvh(shadow_ray, bvh_leaves, nodes, s_cached, verts,
                               node_num, cache_num, emit_len_mis - EPSILON)) {
            // MIS for BSDF / light sampling, to achieve better rendering
            // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf +
            // material_pdf), divided by direct_pdf
            emit_len_mis =
                direct_pdf + c_material[material_id]->pdf(it, shadow_ray.d,
                                                          ray.d, material_id) *
                                 emitter->non_delta();
            rdc += thp * direct_comp *
                   c_material[material_id]->eval(it, shadow_ray.d, ray.d,
                                                 material_id) *
                   (float(emit_len_mis > EPSILON) *
                    __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
            // numerical guard, in case emit_len_mis is 0
        }
        // emitter MIS
        float pdf = payloads.pdf(gidx),
              emission_weight =
                  pdf / (pdf + objects[object_id].solid_angle_pdf(
                                   c_textures.eval_normal(it, material_id),
                                   ray.d, ray.hit_t) *
                                   (emitter_id > 0) * secondary_bounce *
                                   ray.non_delta());
        // (2) account for emission, and accumulate to payload buffer
        payloads.L(gidx) = (thp * c_emitter[emitter_id]->eval_le(&ray.d, &it)) *
                               emission_weight +
                           rdc;
        payloads.set_sampler(gidx, sg);
    }
}

/**
 * @brief This kernel handles path guiding. Here
 * NASG [SIGGRAPH 2024] paper is reproduced (TODO)
 * For this kernel, we will wait until the neural network
 * outputs the evaluation
 *
 * This is the first half of the ray scatter shader, where we need to
 * output the samples that need NN evaluation. For each pixel, we only need to
 * output all the samples to a global pool leveraging multi-stream, and each
 * stream outputs only part of the pool so that we do not need the entire pool
 * to be ready to start NN evaluation
 *
 * For example, we have 16 non blocking streams, and the global pool has 16
 * patches, each thread has 128 threads and 128 blocks, totaling up to a 128 *
 * 128 patch. After generating the 128 * 128 patch, there will be a neural
 * network that use 128 * 128 as batch size for evaluation (no grad). After the
 * result is returned (and consumed), the current stream will move forward to
 * process one more patch (so a for loop is needed).
 */
CPT_KERNEL void guided_ray_scatter_net_eval_shader(
    PayLoadBufferSoA payloads, ConstObjPtr objects,
    const cudaTextureObject_t bvh_leaves, const IndexBuffer idx_buffer,
    int stream_id
    /**TODO: input (evaluated NASG params) and output (training samples) */
) {
    uint8_t ray_stat = 0;
    uint32_t gmem_addr = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t gidx = get_index(idx_buffer, gmem_addr, ray_stat);

    if (ray_stat < 0x80) {
        Vec4 thp = payloads.thp(gidx);
        Ray ray = payloads.get_ray(gidx);
        Sampler sg = payloads.get_sampler(gidx);
        const Interaction it = payloads.interaction(gidx);

        int object_id = tex1Dfetch<int>(bvh_leaves, ray.hit_id());
        object_id = object_id >= 0
                        ? object_id
                        : -object_id - 1; // sphere object ID is -id - 1
        int material_id = 0, emitter_id = 0;
        objects[object_id].unpack(material_id, emitter_id);

        float pdf = 1.0f;

        ScatterStateFlag sampled_lobe = ScatterStateFlag::BSDF_NONE;
        ray.d = c_material[material_id]->sample_dir(ray.d, it, thp, pdf, sg,
                                                    sampled_lobe, material_id);
        ray.set_delta((sampled_lobe & ScatterStateFlag::BSDF_SPECULAR) > 0);

        payloads.thp(gidx) = thp;
        payloads.set_sampler(gidx, sg);
        payloads.set_ray(gidx, ray);
        payloads.pdf(gidx) = pdf;
    }
}

template <bool render_once>
CPT_KERNEL void radiance_splat(PayLoadBufferSoA payloads, DeviceImage image,
                               float *__restrict__ output_buffer,
                               float *__restrict__ var_buffer, int accum_cnt,
                               bool gamma_corr) {

    const uint32_t px = threadIdx.x + blockIdx.x * blockDim.x,
                   py = threadIdx.y + blockIdx.y * blockDim.y;
    const uint32_t gmem_addr = px + py * image.w();

    Vec4 L = payloads.L(gmem_addr); // To local register
    L = L.numeric_err() ? Vec4(0, 0, 0, 1) : L;

    if constexpr (render_once) {
        // image will be the output buffer, there will be double buffering
        auto local_v = image(px, py);
        if (var_buffer)
            estimate_variance(var_buffer, local_v, L, px, py, image.w(),
                              accum_cnt);
        local_v += L;
        image(px, py) = local_v;
        local_v *= 1.f / float(accum_cnt);
        local_v = gamma_corr ? local_v.gamma_corr() : local_v;
        FLOAT4(output_buffer[gmem_addr << 2]) = float4(local_v);
    } else {
        image(px, py) += L;
    }
}

template CPT_KERNEL void radiance_splat<true>(PayLoadBufferSoA payloads,
                                              DeviceImage image,
                                              float *__restrict__ output_buffer,
                                              float *__restrict__ var_buffer,
                                              int accum_cnt, bool gamma_corr);

template CPT_KERNEL void
radiance_splat<false>(PayLoadBufferSoA payloads, DeviceImage image,
                      float *__restrict__ output_buffer,
                      float *__restrict__ var_buffer, int accum_cnt,
                      bool gamma_corr);
