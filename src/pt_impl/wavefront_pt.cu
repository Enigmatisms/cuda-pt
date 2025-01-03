/**
 * Wavefront Path Tracing (Implementation)
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#include "renderer/wavefront_pt.cuh"

namespace {
    using PayLoadBuffer      = PayLoadBufferSoA* const __restrict__;
    using ConstPayLoadBuffer = const PayLoadBuffer;
}

static constexpr int SEED_SCALER = 11451;       //-4!
static constexpr int RR_BOUNCE = 2;
static constexpr float RR_THRESHOLD = 0.1;

/**
 * @brief ray generation kernel 
 * note that, all the kernels are called per stream, each stream can have multiple blocks (since it is a kernel call)
 * let's say, for example, a 4 * 4 block for one kernel call. These 16 blocks should be responsible for 
 * one image patch, offseted by the stream_offset.
 * @note we first consider images that have width and height to be the multiple of 128
 * to avoid having to consider the border problem
 * @note we pass payloads in by value
*/ 
CPT_KERNEL void raygen_primary_hit_shader(
    const DeviceCamera& dev_cam,
    PayLoadBufferSoA payloads,
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const IndexBuffer idx_buffer,
    int stream_offset, int num_prims,
    int x_patch, int y_patch, int iter,
    int stream_id, int width, 
    int node_num, int cache_num
) {
    // stream and patch related offset
    const int sx = x_patch * PATCH_X, sy = y_patch * PATCH_Y, buffer_xoffset = stream_id * PATCH_X;
    // within kernel thread position
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    // linear idx_buffer position
    const int block_index = py * blockDim.x * gridDim.x + px;

    Sampler sg = Sampler(px + sx + (py + sy) * width, iter * SEED_SCALER);
    Ray ray = dev_cam.generate_ray(px + sx, py + sy, sg.next2D());

    Interaction it;                          // To local register

    int min_index = -1, min_object_id = 0;   // round up
    ray.hit_t = MAX_DIST;

    #ifdef FUSED_MISS_SHADER
    ray.set_active(false);
    #endif   // FUSED_MISS_SHADER
    float prim_u = 0, prim_v = 0;

    payloads.thp(px + buffer_xoffset, py) = Vec4(1, 1, 1, 1);
    idx_buffer[block_index + stream_id * TOTAL_RAY] = (py << 16) + px + buffer_xoffset;    
        // cache near root level BVH nodes for faster traversal
    extern __shared__ float4 s_cached[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < 2 * cache_num) {      // no more than 128 nodes will be cached
        s_cached[tid] = cached_nodes[tid];
    }
    __syncthreads();
    ray.hit_t = ray_intersect_bvh(ray, bvh_leaves, nodes, 
                    s_cached, verts, min_index, min_object_id, 
                    prim_u, prim_v, node_num, cache_num, MAX_DIST);

    // ============= step 2: local shading for indirect bounces ================
    payloads.L(px + buffer_xoffset, py)   = Vec4(0, 0, 0, 1);
    payloads.set_sampler(px + buffer_xoffset, py, sg);
    if (min_index >= 0) {
        // if the ray hits nothing, or the path throughput is 0, then the ray will be inactive
        // inactive rays will only be processed in the miss_shader
        ray.set_hit();
        ray.set_hit_index(min_index);
#ifdef FUSED_MISS_SHADER
        ray.set_active(true);
#endif   // FUSED_MISS_SHADER
        it = Primitive::get_interaction(verts, norms, uvs, ray.advance(ray.hit_t), prim_u, prim_v, min_index, min_object_id >= 0);
    }

    // compress two int (to int16) to a uint32_t 
    // note that we can not use int here, since int shifting might retain the sign
    // it is implementation dependent
    // note that we only have stream_number * payloadbuffers
    // so row indices won't be offset by sy, col indices should only be offseted by stream_offset
    payloads.set_ray(px + buffer_xoffset, py, ray);
    payloads.interaction(px + buffer_xoffset, py) = it;
     
    // px has already encoded stream_offset (stream_id * PATCH_X)
}

/**
 * @brief find ray intersection for next hit pos
 * We first start with small pool size (4096), which can comprise at most 16 blocks
 * The ray pool is stream-compacted (with thrust::parition to remove the finished)
 * Note that we need an index buffer, since the Ray and Sampler are coupled
 * and we need the index to port the 
*/ 
CPT_KERNEL void closesthit_shader(
    PayLoadBufferSoA payloads,
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims,
    int num_valid,
    int node_num,
    int cache_num
) {
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px

    uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
    py >>= 16;
    Ray        ray = payloads.get_ray(px, py);
    Interaction it = payloads.interaction(px, py);                           // To local register
    ray.reset();
    
    float prim_u = 0, prim_v = 0;
    int min_index = -1, min_object_id = 0;   // round up
    ray.hit_t = MAX_DIST;

    // cache near root level BVH nodes for faster traversal
    extern __shared__ float4 s_cached[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < 2 * cache_num) {      // no more than 128 nodes will be cached
        s_cached[tid] = cached_nodes[tid];
    }
    __syncthreads();
    ray.hit_t = ray_intersect_bvh(ray, bvh_leaves, nodes, 
                    s_cached, verts, min_index, min_object_id, 
                    prim_u, prim_v, node_num, cache_num, MAX_DIST);

    // ============= step 2: local shading for indirect bounces ================
    if (block_index < num_valid && min_index >= 0) {
        // if the ray hits nothing, or the path throughput is 0, then the ray will be inactive
        // inactive rays will only be processed in the miss_shader
        ray.set_hit();
        ray.set_hit_index(min_index);
#ifdef FUSED_MISS_SHADER
        ray.set_active(true);
#endif   // FUSED_MISS_SHADER
        it = Primitive::get_interaction(verts, norms, uvs, ray.advance(ray.hit_t), prim_u, prim_v, min_index, min_object_id >= 0);
    }

    payloads.set_ray(px, py, ray);
    payloads.interaction(px, py) = it;
}

/***
 * For non-delta hit (shading point), direct component should be evaluated:
 * we sample a light source then start ray intersection test
*/
CPT_KERNEL void nee_shader(
    PayLoadBufferSoA payloads,
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims,
    int num_objects,
    int num_emitter,
    int num_valid,
    int node_num,
    int cache_num
) {
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px
    // cache near root level BVH nodes for faster traversal
    extern __shared__ float4 s_cached[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < 2 * cache_num) {      // no more than 128 nodes will be cached
        s_cached[tid] = cached_nodes[tid];
    }
    __syncthreads();
    
    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
        py >>= 16;
        Vec4 thp = payloads.thp(px, py);
        Ray ray  = payloads.get_ray(px, py);
        Sampler sg = payloads.get_sampler(px, py);
        const Interaction it = payloads.interaction(px, py);

        int object_id = tex1Dfetch<int>(bvh_leaves, ray.hit_id());
        object_id = object_id >= 0 ? object_id : -object_id - 1;        // sphere object ID is -id - 1
        int material_id = 0, emitter_id = -1;
            objects[object_id].unpack(material_id, emitter_id);

        float direct_pdf = 1;

        Emitter* emitter = sample_emitter(sg, direct_pdf, num_emitter, emitter_id);
        emitter_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(sg.discrete1D(), direct_pdf);
        emitter_id = emitter_prims[emitter_id];               // extra mapping, introduced after BVH primitive reordering
        Ray shadow_ray(ray.advance(ray.hit_t), Vec3(0, 0, 0));
        // use ray.o to avoid creating another shadow_int variable
        Vec4 direct_comp(0, 0, 0, 1);
        shadow_ray.d = emitter->sample(shadow_ray.o, it.shading_norm, direct_comp, direct_pdf, sg.next2D(), verts, norms, uvs, emitter_id) - shadow_ray.o;

        float emit_len_mis = shadow_ray.d.length();
        shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direct
        // (3) NEE scene intersection test (possible warp divergence, but... nevermind)
        if (emitter != c_emitter[0] && 
            occlusion_test_bvh(shadow_ray, bvh_leaves, nodes, 
                    s_cached, verts, node_num, cache_num, emit_len_mis - EPSILON)
        ) {
            // MIS for BSDF / light sampling, to achieve better rendering
            // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
            emit_len_mis = direct_pdf + c_material[material_id]->pdf(it, shadow_ray.d, ray.d, material_id) * emitter->non_delta();
            payloads.L(px, py) += thp * direct_comp * c_material[material_id]->eval(it, shadow_ray.d, ray.d, material_id) * \
                (float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
            // numerical guard, in case emit_len_mis is 0
        }

        payloads.set_sampler(px, py, sg);
    }
}

/**
 * BSDF sampling & direct shading shader
*/
CPT_KERNEL void bsdf_local_shader(
    PayLoadBufferSoA payloads,
    const cudaTextureObject_t bvh_leaves,
    ConstObjPtr objects,
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims, 
    int num_valid,
    bool secondary_bounce
) {
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px

    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
        py >>= 16;

        Vec4 thp = payloads.thp(px, py);
        Ray ray  = payloads.get_ray(px, py);
        Sampler sg = payloads.get_sampler(px, py);
        Interaction it = payloads.interaction(px, py);

        // this is incorrect, since AABB should be reordered, too
        int object_id = tex1Dfetch<int>(bvh_leaves, ray.hit_id());
        object_id = object_id >= 0 ? object_id : -object_id - 1;        // sphere object ID is -id - 1
        int material_id = 0, emitter_id = -1;
            objects[object_id].unpack(material_id, emitter_id);
        
        bool hit_emitter = emitter_id > 0;

        // emitter MIS
        float pdf = payloads.pdf(px, py), emission_weight = pdf / (pdf + 
                objects[object_id].solid_angle_pdf(c_textures.eval_normal(it, material_id), ray.d, ray.hit_t) * 
                hit_emitter * secondary_bounce * ray.non_delta());
        // (2) check if the ray hits an emitter
        Vec4 direct_comp = thp *\
                    c_emitter[emitter_id]->eval_le(&ray.d, &it);
        payloads.L(px, py) += direct_comp * emission_weight;
        
        ray.o = ray.advance(ray.hit_t);
        BSDFFlag sampled_lobe = BSDFFlag::BSDF_NONE;                            
        ray.d = c_material[material_id]->sample_dir(
            ray.d, it, thp, pdf, sg, sampled_lobe, material_id
        );
        ray.set_delta((sampled_lobe & BSDFFlag::BSDF_SPECULAR) > 0);

        payloads.set_sampler(px, py, sg);
        payloads.thp(px, py) = thp;
        payloads.set_ray(px, py, ray);
        payloads.interaction(px, py) = it;
        payloads.pdf(px, py) = pdf;
    }
}

/**
 * Purpose of the miss shader: if ray hits nothing in closesthit shader
 * the we will set the hit status (flag) to be false
 * in this shader, we find the rays marked as no-hit, and check the
 * availability of environment map (currently not supported)
 * after processing the env-map lighting, we mark the ray as inactive
 * before stream compaction. Then stream compaction will 'remove' all these
 * rays (and the threads)
 * 
 * MISS_SHADER is the only place where you mark a ray as inactive
*/
CPT_KERNEL void miss_shader(
    PayLoadBufferSoA payloads,
    const IndexBuffer idx_buffer,
    const int bounce,
    int stream_offset,
    int num_valid,
    int envmap_id
) {
    // Nothing here, currently, if we decide not to support env lighting
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px
    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
        py >>= 16;
        Vec4 thp        = payloads.thp(px, py), old_thp = thp;
        Sampler sampler = payloads.get_sampler(px, py);
        // using BVH enables the usage of RR, since there is no within-loop synchronization
        float max_value = thp.max_elem_3d();

        if (bounce >= RR_BOUNCE && max_value < RR_THRESHOLD) {
            max_value = (sampler.next1D() > max_value || max_value < THP_EPS) ? 0 : max_value;
            thp *= max_value == 0 ? 0 : (1.f / max_value);
        }

        Vec3 ray_d;
        bool is_active = true;
        payloads.get_ray_d(px, py, ray_d, is_active);
        if (is_active && !payloads.is_hit(px, py)) {
            payloads.L(px, py) += old_thp * c_emitter[envmap_id]->eval_le(&ray_d);
            payloads.set_active(px, py, false);
        } else if (max_value <= 1e-5f) {
            payloads.set_active(px, py, false);
        }
        payloads.thp(px, py) = thp;
        payloads.set_sampler(px, py, sampler);
    }
}

template <bool render_once>
CPT_KERNEL void radiance_splat(
    PayLoadBufferSoA payloads, DeviceImage image, 
    int stream_id, int x_patch, int y_patch, 
    int accum_cnt, float* output_buffer, bool gamma_corr
) {
    // Nothing here, currently, if we decide not to support env lighting
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    Vec4 L = payloads.L(px + stream_id * PATCH_X, py);         // To local register
    L = L.numeric_err() ? Vec4(0, 0, 0, 1) : L;

    if constexpr (render_once) {
        // image will be the output buffer, there will be double buffering
        int img_x = px + x_patch * PATCH_X, img_y = py + y_patch * PATCH_Y;
        auto local_v = image(img_x, img_y) + L;
        image(img_x, img_y) = local_v;
        local_v *= 1.f / float(accum_cnt);
        local_v = gamma_corr ? local_v.gamma_corr() : local_v;
        FLOAT4(output_buffer[(img_x + img_y * image.w()) << 2]) = float4(local_v); 
    } else {
        image(px + x_patch * PATCH_X, py + y_patch * PATCH_Y) += L;
    }
}

template CPT_KERNEL void radiance_splat<true>(
    PayLoadBufferSoA payloads, DeviceImage image, 
    int stream_id, int x_patch, int y_patch,
    int accum_cnt, float* output_buffer, bool gamma_corr
);

template CPT_KERNEL void radiance_splat<false>(
    PayLoadBufferSoA payloads, DeviceImage image, 
    int stream_id, int x_patch, int y_patch, 
    int accum_cnt, float* output_bufferr, bool gamma_corr
);