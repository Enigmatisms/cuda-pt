/**
 * Wavefront Path Tracing (Implementation)
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#include "renderer/base_pt.cuh"
#include "renderer/wavefront_pt.cuh"

namespace {
    using PayLoadBuffer      = PayLoadBufferSoA* const;
    using ConstPayLoadBuffer = const PayLoadBuffer;
}

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
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    const IndexBuffer idx_buffer,
    int stream_offset, int num_prims,
    int x_patch, int y_patch, int iter,
    int stream_id, int width, int node_num
) {
    // stream and patch related offset
    const int sx = x_patch * PATCH_X, sy = y_patch * PATCH_Y, buffer_xoffset = stream_id * PATCH_X;
    // within kernel thread position
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    // linear idx_buffer position
    const int block_index = py * blockDim.x * gridDim.x + px;

    Sampler sg = Sampler(px + sx + (py + sy) * width, iter * SEED_SCALER);
    Ray ray = dev_cam.generate_ray(px + sx, py + sy, sg.next2D());

    PDFInteraction it;            // To local register
    ShapeExtractVisitor extract(*verts, *norms, *uvs, ray, 0);

    int min_index = -1;   // round up
    ray.hit_t = MAX_DIST;

    #ifdef FUSED_MISS_SHADER
    ray.set_active(false);
    #endif   // FUSED_MISS_SHADER

    payloads.thp(px + buffer_xoffset, py) = Vec4(1, 1, 1, 1);
    idx_buffer[block_index + stream_id * TOTAL_RAY] = (py << 16) + px + buffer_xoffset;    
#ifdef RENDERER_USE_BVH 
    ShapeIntersectVisitor visitor(*verts, ray, 0);
    ray.hit_t = ray_intersect_bvh(ray, shapes, bvh_fronts, bvh_backs, node_fronts, 
                    node_backs, node_offsets, visitor, min_index, node_num, ray.hit_t);
#else   // RENDERER_USE_BVH
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ Vec3 s_verts[TRI_IDX(BASE_ADDR)];                // vertex info
    __shared__ AABBWrapper s_aabbs[BASE_ADDR];                  // aabb

    ArrayType<Vec3> s_verts_arr(reinterpret_cast<Vec3*>(&s_verts[0]), BASE_ADDR);
    ShapeIntersectVisitor visitor(s_verts_arr, ray, 0);
    int num_copy = (num_prims + BASE_ADDR - 1) / BASE_ADDR;

    // ============= step 1: ray intersection =================
    #pragma unroll
    for (int cp_base = 0; cp_base < num_copy; ++cp_base) {
        // memory copy to shared memory
        int cur_idx = (cp_base << BASE_SHFL) + tid, remain_prims = min(num_prims - (cp_base << BASE_SHFL), BASE_ADDR);
        cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
        if (tid < BASE_ADDR && cur_idx < num_prims) {        // copy from gmem to smem
    #ifdef USE_SOA          // SOA is actually better
            cuda::memcpy_async(&s_verts[tid],                    &verts->x(cur_idx), sizeof(Vec3), pipe);
            cuda::memcpy_async(&s_verts[tid + BASE_ADDR],        &verts->y(cur_idx), sizeof(Vec3), pipe);
            cuda::memcpy_async(&s_verts[tid + (BASE_ADDR << 1)], &verts->z(cur_idx), sizeof(Vec3), pipe);
    #else
            // we should pad this, for every 3 Vec3, we pad one more vec3, then copy can be made
            // without branch (memcpy_async): TODO, this is the bottle neck. L2 Global excessive here
            // since our step is Vec3, this will lead to uncoalesced access
            // shared memory is enough. Though padding is not easy to implement
            cuda::memcpy_async(&s_verts[TRI_IDX(tid)], &verts->data[TRI_IDX(cur_idx)], sizeof(Vec3) * 3, pipe);
    #endif
            // This memory op is not fully coalesced, since AABB container is not a complete SOA
            s_aabbs[tid].aabb.copy_from(aabbs[cur_idx]);
        }
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();
        // this might not be a good solution
        ray.hit_t = ray_intersect(ray, shapes, s_aabbs, visitor, min_index, remain_prims, cp_base << BASE_SHFL, ray.hit_t);
        __syncthreads();
    }
#endif  // RENDERER_USE_BVH
    // ============= step 2: local shading for indirect bounces ================
    payloads.L(px + buffer_xoffset, py)   = Vec4(0, 0, 0, 1);
    payloads.set_sampler(px + buffer_xoffset, py, sg);
    if (min_index >= 0) {
        // if the ray hits nothing, or the path throughput is 0, then the ray will be inactive
        // inactive rays will only be processed in the miss_shader
        extract.set_index(min_index);
        ray.set_hit();
        ray.set_hit_index(min_index);
#ifdef FUSED_MISS_SHADER
        ray.set_active(true);
#endif   // FUSED_MISS_SHADER
        it.it() = variant::apply_visitor(extract, shapes[min_index]);
    }

    // compress two int (to int16) to a uint32_t 
    // note that we can not use int here, since int shifting might retain the sign
    // it is implementation dependent
    // note that we only have stream_number * payloadbuffers
    // so row indices won't be offset by sy, col indices should only be offseted by stream_offset
    payloads.set_ray(px + buffer_xoffset, py, ray);
    payloads.set_interaction(px + buffer_xoffset, py, it);
     
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
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims,
    int num_valid,
    int node_num
) {
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px
                            
    
    uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
    py >>= 16;
    Ray           ray = payloads.get_ray(px, py);
    PDFInteraction it = payloads.get_interaction(px, py);            // To local register
    ray.reset();
    
    ShapeExtractVisitor extract(*verts, *norms, *uvs, ray, 0);

    int min_index = -1;   // round up
    ray.hit_t = MAX_DIST;

#ifdef RENDERER_USE_BVH 
    ShapeIntersectVisitor visitor(*verts, ray, 0);
    ray.hit_t = ray_intersect_bvh(ray, shapes, bvh_fronts, bvh_backs, 
                    node_fronts, node_backs, node_offsets, visitor, min_index, node_num, ray.hit_t);
#else   // RENDERER_USE_BVH
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ Vec3 s_verts[TRI_IDX(BASE_ADDR)];                // vertex info
    __shared__ AABBWrapper s_aabbs[BASE_ADDR];                  // aabb

    ArrayType<Vec3> s_verts_arr(reinterpret_cast<Vec3*>(&s_verts[0]), BASE_ADDR);
    ShapeIntersectVisitor visitor(s_verts_arr, ray, 0);
    int num_copy = (num_prims + BASE_ADDR - 1) / BASE_ADDR;

    // ============= step 1: ray intersection =================
    #pragma unroll
    for (int cp_base = 0; cp_base < num_copy; ++cp_base) {
        // memory copy to shared memory
        int cur_idx = (cp_base << BASE_SHFL) + tid, remain_prims = min(num_prims - (cp_base << BASE_SHFL), BASE_ADDR);
        cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
        if (tid < BASE_ADDR && cur_idx < num_prims) {        // copy from gmem to smem
    #ifdef USE_SOA          // SOA is actually better
            cuda::memcpy_async(&s_verts[tid],                    &verts->x(cur_idx), sizeof(Vec3), pipe);
            cuda::memcpy_async(&s_verts[tid + BASE_ADDR],        &verts->y(cur_idx), sizeof(Vec3), pipe);
            cuda::memcpy_async(&s_verts[tid + (BASE_ADDR << 1)], &verts->z(cur_idx), sizeof(Vec3), pipe);
    #else
            // we should pad this, for every 3 Vec3, we pad one more vec3, then copy can be made
            // without branch (memcpy_async): TODO, this is the bottle neck. L2 Global excessive here
            // since our step is Vec3, this will lead to uncoalesced access
            // shared memory is enough. Though padding is not easy to implement
            cuda::memcpy_async(&s_verts[TRI_IDX(tid)], &verts->data[TRI_IDX(cur_idx)], sizeof(Vec3) * 3, pipe);
    #endif
            // This memory op is not fully coalesced, since AABB container is not a complete SOA
            s_aabbs[tid].aabb.copy_from(aabbs[cur_idx]);
        }
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();
        // this might not be a good solution
        ray.hit_t = ray_intersect(ray, shapes, s_aabbs, visitor, min_index, remain_prims, cp_base << BASE_SHFL, ray.hit_t);
        __syncthreads();
    }
#endif  // RENDERER_USE_BVH

    // ============= step 2: local shading for indirect bounces ================
    if (block_index < num_valid && min_index >= 0) {
        // if the ray hits nothing, or the path throughput is 0, then the ray will be inactive
        // inactive rays will only be processed in the miss_shader
        extract.set_index(min_index);
        ray.set_hit();
        ray.set_hit_index(min_index);
#ifdef FUSED_MISS_SHADER
        ray.set_active(true);
#endif   // FUSED_MISS_SHADER
        it.it() = variant::apply_visitor(extract, shapes[min_index]);
    }

    payloads.set_ray(px, py, ray);
    payloads.set_interaction(px, py, it);
}

/***
 * For non-delta hit (shading point), direct component should be evaluated:
 * we sample a light source then start ray intersection test
*/
CPT_KERNEL void nee_shader(
    PayLoadBufferSoA payloads,
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr,         
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims,
    int num_objects,
    int num_emitter,
    int num_valid,
    int node_num
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
        const PDFInteraction it = payloads.get_interaction(px, py);

        int object_id    = prim2obj[ray.hit_id()],
            material_id  = objects[object_id].bsdf_id,
            emitter_id   = objects[object_id].emitter_id;

        float direct_pdf = 1;

        Emitter* emitter = sample_emitter(sg, direct_pdf, num_emitter, emitter_id);
        emitter_id       = objects[emitter->get_obj_ref()].sample_emitter_primitive(sg.discrete1D(), direct_pdf);
        Ray shadow_ray(ray.advance(ray.hit_t), Vec3(0, 0, 0));
        // use ray.o to avoid creating another shadow_int variable
        Vec4 direct_comp(0, 0, 0, 1);
        shadow_ray.d = emitter->sample(shadow_ray.o, direct_comp, direct_pdf, sg.next2D(), verts, norms, emitter_id) - shadow_ray.o;

        float emit_len_mis = shadow_ray.d.length();
        shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direct
        // (3) NEE scene intersection test (possible warp divergence, but... nevermind)
        if (emitter != c_emitter[0] && 
#ifdef RENDERER_USE_BVH
            occlusion_test_bvh(shadow_ray, shapes, bvh_fronts, bvh_backs, 
                        node_fronts, node_backs, node_offsets, *verts, node_num, emit_len_mis - EPSILON)
#else   // RENDERER_USE_BVH
            occlusion_test(shadow_ray, objects, shapes, aabbs, *verts, num_objects, emit_len_mis - EPSILON)
#endif  // RENDERER_USE_BVH
        ) {
            // MIS for BSDF / light sampling, to achieve better rendering
            // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
            emit_len_mis = direct_pdf + c_material[material_id]->pdf(it.it_const(), shadow_ray.d, ray.d) * emitter->non_delta();
            payloads.L(px, py) += thp * direct_comp * c_material[material_id]->eval(it.it_const(), shadow_ray.d, ray.d) * \
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
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstUVPtr,         
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
        PDFInteraction it = payloads.get_interaction(px, py);
        Vec2 sample = sg.next2D();
        payloads.set_sampler(px, py, sg);

        int object_id   = prim2obj[ray.hit_id()],
            emitter_id  = objects[object_id].emitter_id,
            material_id = objects[object_id].bsdf_id;
        bool hit_emitter = emitter_id > 0;

        // emitter MIS
        float emission_weight = it.pdf_v() / (it.pdf_v() + 
                objects[object_id].solid_angle_pdf(it.it_const().shading_norm, ray.d, ray.hit_t) * hit_emitter * secondary_bounce);
        // (2) check if the ray hits an emitter
        Vec4 direct_comp = thp *\
                    c_emitter[emitter_id]->eval_le(&ray.d, &it.it_const().shading_norm);
        payloads.L(px, py) += direct_comp * emission_weight;
        
        ray.o = ray.advance(ray.hit_t);
        ray.d = c_material[material_id]->sample_dir(
            ray.d, it.it_const(), thp, it.pdf(), std::move(sample)
        );

        payloads.thp(px, py) = thp;
        payloads.set_ray(px, py, ray);
        payloads.set_it_head(px, py, it.data.p1);
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
    int stream_offset,
    int num_valid
) {
    // Nothing here, currently, if we decide not to support env lighting
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px
    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
        py >>= 16;
        Vec4 thp = payloads.thp(px, py);
        if ((!payloads.is_hit(px, py)) || thp.max_elem() <= 1e-5f) {
            // TODO: process no-hit ray, environment map lighting
            payloads.set_active(px, py, false);
        }
    }
}

CPT_KERNEL void radiance_splat(
    PayLoadBufferSoA payloads, DeviceImage& image, 
    int stream_id, int x_patch, int y_patch
) {
    // Nothing here, currently, if we decide not to support env lighting
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    Vec4 L = payloads.L(px + stream_id * PATCH_X, py);         // To local register
    image(px + x_patch * PATCH_X, py + y_patch * PATCH_Y) += L.numeric_err() ? Vec4(0, 0, 0, 1) : L;
}