/**
 * Megakernel Light Tracing (Implementation)
 * Note that though LT and PT are both declared in `megakernel_pt.cuh`
 * We separate LT and PT implementation, for the sake of clarity
 * Also, I only intend to implement the megakernel version
 * Since LT is not so tile-based, WF ideas are less intuitive
 * just for me
 * 
 * @date: 9.28.2024
 * @author: Qianyue He
*/
#include "renderer/base_pt.cuh"
#include "renderer/megakernel_pt.cuh"

static constexpr int RR_BOUNCE = 2;
static constexpr float RR_THRESHOLD = 0.1;

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level culling
 * shared memory might not be easy to use, since the memory granularity will be
 * too difficult to control
 * 
 * @param objects   object encapsulation
 * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
 * @param camera    GPU camera model (constant memory)
 * @param image     GPU image buffer
 * @param num_prims number of primitives (to be intersected with)
 * @param max_depth maximum allowed bounce
*/
template <bool render_once>
CPT_KERNEL void render_lt_kernel(
    const DeviceCamera& dev_cam, 
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    const cudaTextureObject_t bvh_fronts,
    const cudaTextureObject_t bvh_backs,
    const cudaTextureObject_t node_fronts,
    const cudaTextureObject_t node_backs,
    const cudaTextureObject_t node_offsets,
    DeviceImage& image,
    float* output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth,/* max depth, useless for depth renderer, 1 anyway */
    int node_num,
    int accum_cnt,
    int specular_constraints,
    float caustic_scale
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    int constraint_cnt = 0;

    Sampler sampler(px + py * image.w(), seed_offset);
    // step 1: generate ray (sample ray from one emitter and sample from that emitter)
    Ray ray;
    Vec4 throughput;
    {
        uint32_t emitter_id = (sampler.discrete1D() % uint32_t(num_emitter)) + 1;
        Emitter* emitter = c_emitter[emitter_id];
        float emitter_sample_pdf = 1.f / float(num_emitter), le_pdf = 1.f;

        Vec2 extras = sampler.next2D();
        emitter_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(sampler.discrete1D(), le_pdf);
        throughput = emitter->sample_le(ray.o, ray.d, le_pdf, sampler.next2D(), verts, norms, emitter_id, extras.x(), extras.y());
        throughput *= 1.f / (emitter_sample_pdf * le_pdf);
    }

    // step 2: bouncing around the scene until the max depth is reached
    int min_index = -1, object_id = 0;
#ifndef RENDERER_USE_BVH
    __shared__ Vec3 s_verts[TRI_IDX(BASE_ADDR)];         // vertex info
    __shared__ AABBWrapper s_aabbs[BASE_ADDR];            // aabb
    ArrayType<Vec3> s_verts_arr(reinterpret_cast<Vec3*>(&s_verts[0]), BASE_ADDR);
    int num_copy = (num_prims + BASE_ADDR - 1) / BASE_ADDR;   // round up
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
#endif  // RENDERER_USE_BVH
    for (int b = 0; b < max_depth; b++) {
        float prim_u = 0, prim_v = 0, min_dist = MAX_DIST;
        min_index = -1;
        // ============= step 1: ray intersection =================
#ifdef RENDERER_USE_BVH
        min_dist = ray_intersect_bvh(
            ray, bvh_fronts, bvh_backs, 
            node_fronts, node_backs, node_offsets, 
            *verts, min_index, object_id, prim_u, 
            prim_v, node_num, min_dist
        );
#else   // RENDERER_USE_BVH
        #pragma unroll
        for (int cp_base = 0; cp_base < num_copy; ++cp_base) {
            // memory copy to shared memory
            int cur_idx = (cp_base << BASE_SHFL) + tid, remain_prims = min(num_prims - (cp_base << BASE_SHFL), BASE_ADDR);
            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

            // huge bug
            if (tid < BASE_ADDR && cur_idx < num_prims) {        // copy from gmem to smem
    #ifdef USE_SOA
                cuda::memcpy_async(&s_verts[tid],                    &verts->x(cur_idx), sizeof(Vec3), pipe);
                cuda::memcpy_async(&s_verts[tid + BASE_ADDR],        &verts->y(cur_idx), sizeof(Vec3), pipe);
                cuda::memcpy_async(&s_verts[tid + (BASE_ADDR << 1)], &verts->z(cur_idx), sizeof(Vec3), pipe);
    #else
                cuda::memcpy_async(&s_verts[TRI_IDX(tid)], &verts->data[TRI_IDX(cur_idx)], sizeof(Vec3) * 3, pipe);
    #endif
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
            // printf("Throughput: %f, %f, %f, min_index: %d, bounce: %d, %d\n", throughput.x(), throughput.y(), throughput.z(), min_index, b, num_emitter);
            auto it = Primitive::get_interaction(*verts, *norms, *uvs, ray.advance(min_dist), prim_u, prim_v, min_index, object_id >= 0);
            object_id = object_id >= 0 ? object_id : -object_id - 1;        // sphere object ID is -id - 1

            // ============= step 3: next event estimation ================
            // (1) randomly pick one emitter
            int material_id = objects[object_id].bsdf_id;

            // deterministically connect to the camera
            Ray shadow_ray(ray.advance(min_dist), Vec3(0, 0, 1));
            shadow_ray.d = dev_cam.t - shadow_ray.o;
            float emit_len_mis = shadow_ray.d.length();
            shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direction

            // (3) Light tracing NEE scene intersection test (possible warp divergence, but... nevermind)
            int pixel_x = -2, pixel_y = -2;
            if (constraint_cnt > specular_constraints &&
                dev_cam.get_splat_pixel(shadow_ray.d, pixel_x, pixel_y) && 
#ifdef RENDERER_USE_BVH
            occlusion_test_bvh(shadow_ray, bvh_fronts, bvh_backs, node_fronts, 
                        node_backs, node_offsets, *verts, node_num, emit_len_mis - EPSILON)
#else   // RENDERER_USE_BVH
            occlusion_test(shadow_ray, objects, aabbs, *verts, num_objects, emit_len_mis - EPSILON)
#endif  // RENDERER_USE_BVH
            ) {
                Vec4 direct_splat = throughput * c_material[material_id]->eval(it, shadow_ray.d, ray.d) * \
                    (float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
                auto& to_write = image(pixel_x, pixel_y);
                atomicAdd(&to_write.x(), direct_splat.x() * caustic_scale);
                atomicAdd(&to_write.y(), direct_splat.y() * caustic_scale);
                atomicAdd(&to_write.z(), direct_splat.z() * caustic_scale);
                atomicAdd(&to_write.w(), 1.f);
            }

            // step 4: sample a new ray direction, bounce the 
            ray.o = std::move(shadow_ray.o);
            ray.d = c_material[material_id]->sample_dir(ray.d, it, throughput, emit_len_mis, sampler.next2D());
            constraint_cnt += c_material[material_id]->require_lobe(BSDFFlag::BSDF_SPECULAR);

            if (throughput.numeric_err() || throughput < EPSILON) {
                break;
            }

            // step 5: russian roulette
            float max_value = throughput.max_elem_3d();
            if (b >= RR_BOUNCE && max_value < RR_THRESHOLD) {
                if (sampler.next1D() > max_value || max_value < THP_EPS) break;
                throughput *= 1. / max_value;
            }
        }
    }
    __syncthreads();
    if constexpr (render_once) {
        // image will be the output buffer, there will be double buffering
        Vec4 radiance = image(px, py);
        radiance *= 1.f / float(accum_cnt);
        FLOAT4(output_buffer[(px + py * image.w()) << 2]) = float4(radiance); 
    }
}

template CPT_KERNEL void render_lt_kernel<true>(
    const DeviceCamera& dev_cam, 
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    DeviceImage& image,
    float* output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int node_num,
    int accum_cnt,
    int specular_constraints,
    float caustic_scale
);

template CPT_KERNEL void render_lt_kernel<false>(
    const DeviceCamera& dev_cam, 
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    DeviceImage& image,
    float* output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int node_num,
    int accum_cnt,
    int specular_constraints,
    float caustic_scale
);