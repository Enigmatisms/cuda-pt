/**
 * Megakernel Path Tracing (Implementation)
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#include "renderer/base_pt.cuh"
#include "renderer/megakernel_pt.cuh"

/**
 * Occlusion test, computation is done on global memory
*/
CPT_GPU bool occlusion_test(
    const Ray& ray,
    ConstObjPtr objects,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    const ArrayType<Vec3>& verts,
    int num_objects,
    float max_dist
) {
    float aabb_tmin = 0;
    ShapeIntersectVisitor shape_visitor(verts, ray, 0, max_dist);
    int prim_id = 0, num_prim = 0;
    for (int obj_id = 0; obj_id < num_objects; obj_id ++) {
        num_prim = prim_id + objects[obj_id].prim_num;
        if (objects[obj_id].intersect(ray, aabb_tmin) && aabb_tmin < max_dist) {
            // ray intersects the object
            for (; prim_id < num_prim; prim_id ++) {
                if (aabbs[prim_id].intersect(ray, aabb_tmin) && aabb_tmin < max_dist) {
                    shape_visitor.set_index(prim_id);
                    float dist = variant::apply_visitor(shape_visitor, shapes[prim_id]);
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
    ConstShapePtr shapes,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    const ArrayType<Vec3>& verts,
    const int node_num,
    float max_dist
) {
    int node_idx = 0;
    float aabb_tmin = 0;
    // There can be much control flow divergence, not good
    ShapeIntersectVisitor shape_visitor(verts, ray, 0, max_dist);
    while (node_idx < node_num) {
        float4 node_f = tex1Dfetch<float4>(node_fronts, node_idx),
               node_b = tex1Dfetch<float4>(node_backs, node_idx);
        LinearNode node(node_f, node_b);
        int all_offset = tex1Dfetch<int>(node_offsets, node_idx);
        bool intersect_node = node.aabb.intersect(ray, aabb_tmin) && aabb_tmin < max_dist;
        if (!intersect_node) {
            node_idx += all_offset;
            continue;
        }
        if (all_offset == 1) {
            int beg_idx = 0, end_idx = 0;
            node.get_range(beg_idx, end_idx);
            for (int idx = beg_idx; idx < end_idx; idx ++) {
                // if current ray intersects primitive at [idx], tasks will store it
                float4 bvh_f = tex1Dfetch<float4>(bvh_fronts, idx),
                       bvh_b = tex1Dfetch<float4>(bvh_backs,  idx);
                const LinearBVH bvh(bvh_f, bvh_b);
                if (bvh.aabb.intersect(ray, aabb_tmin)) {
                    int obj_idx = 0, prim_idx = 0;
                    bvh.get_info(obj_idx, prim_idx);
                    shape_visitor.set_index(prim_idx);
                    float dist = variant::apply_visitor(shape_visitor, shapes[prim_idx]);
                    if (dist > EPSILON && dist < max_dist)
                        return false;
                }
            }
        }
        node_idx ++;
    }
    return true;
}

/**
 * Stackless BVH (should use tetxure memory?)
 * Perform ray-intersection test on shared memory primitives
 * @param ray: the ray for intersection test
 * @param shapes: scene primitives
 * @param s_aabbs: scene primitives
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
    ConstShapePtr shapes,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    ShapeIntersectVisitor& shape_visitor,
    int& min_index,
    const int node_num,
    float min_dist
) {
    int node_idx = 0;
    float aabb_tmin = 0;
    // There can be much control flow divergence, not good
    while (node_idx < node_num) {
        float4 node_f = tex1Dfetch<float4>(node_fronts, node_idx),
               node_b = tex1Dfetch<float4>(node_backs, node_idx);
        LinearNode node(node_f, node_b);
        int all_offset = tex1Dfetch<int>(node_offsets, node_idx);
        bool intersect_node = node.aabb.intersect(ray, aabb_tmin) && aabb_tmin < min_dist;
        if (!intersect_node) {
            node_idx += all_offset;
            continue;
        }
        if (all_offset == 1) {
            int beg_idx = 0, end_idx = 0;
            node.get_range(beg_idx, end_idx);
            for (int idx = beg_idx; idx < end_idx; idx ++) {
                // if current ray intersects primitive at [idx], tasks will store it
                float4 bvh_f = tex1Dfetch<float4>(bvh_fronts, idx),
                       bvh_b = tex1Dfetch<float4>(bvh_backs,  idx);
                const LinearBVH bvh(bvh_f, bvh_b);
                if (bvh.aabb.intersect(ray, aabb_tmin)) {
                    int obj_idx = 0, prim_idx = 0;
                    bvh.get_info(obj_idx, prim_idx);
                    // we might not need an obj_idx stored in shapes, if we use BVH 
                    shape_visitor.set_index(prim_idx);
                    float dist = variant::apply_visitor(shape_visitor, shapes[prim_idx]);
                    bool valid = dist > EPSILON && dist < min_dist;
                    min_dist = valid ? dist : min_dist;
                    min_index = valid ? prim_idx : min_index;
                }
            }
        }
        node_idx ++;
    }
    return min_dist;
}

CPT_GPU Emitter* sample_emitter(Sampler& sampler, float& pdf, int num, int no_sample) {
    // logic: if no_sample and num > 1, means that there is one emitter than can not be sampled
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

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level culling
 * shared memory might not be easy to use, since the memory granularity will be
 * too difficult to control
 * 
 * @param objects   object encapsulation
 * @param prim2obj  primitive to object index mapping: which object does this primitive come from?
 * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
 * @param camera    GPU camera model (constant memory)
 * @param image     GPU image buffer
 * @param num_prims number of primitives (to be intersected with)
 * @param max_depth maximum allowed bounce
*/
template <bool render_once>
CPT_KERNEL void render_pt_kernel(
    const DeviceCamera& dev_cam, 
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
    DeviceImage& image,
    float* output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth,/* max depth, useless for depth renderer, 1 anyway */
    int node_num,
    int accum_cnt
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;

    Sampler sampler(px + py * image.w(), seed_offset);
    // step 1: generate ray
    Ray ray = dev_cam.generate_ray(px, py, sampler);

    // step 2: bouncing around the scene until the max depth is reached

    // A matter of design choice
    // optimization: copy at most 32 prims from global memory to shared memory

    // this kinda resembles deferred rendering
    int min_index = -1;
#ifdef RENDERER_USE_BVH
    ShapeIntersectVisitor visitor(*verts, ray, 0);
#else   // RENDERER_USE_BVH
    __shared__ Vec3 s_verts[TRI_IDX(BASE_ADDR)];         // vertex info
    __shared__ AABBWrapper s_aabbs[BASE_ADDR];            // aabb
    ArrayType<Vec3> s_verts_arr(reinterpret_cast<Vec3*>(&s_verts[0]), BASE_ADDR);
    ShapeIntersectVisitor visitor(s_verts_arr, ray, 0);
    int num_copy = (num_prims + BASE_ADDR - 1) / BASE_ADDR;   // round up
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
#endif  // RENDERER_USE_BVH
    ShapeExtractVisitor extract(*verts, *norms, *uvs, ray, 0);

    Vec4 throughput(1, 1, 1), radiance(0, 0, 0);
    float emission_weight = 1.f;
    bool hit_emitter = false;
    for (int b = 0; b < max_depth; b++) {
        float min_dist = MAX_DIST;
        min_index = -1;
        // ============= step 1: ray intersection =================
#ifdef RENDERER_USE_BVH
        min_dist = ray_intersect_bvh(ray, shapes, bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets, visitor, min_index, node_num, min_dist);
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
            min_dist = ray_intersect(ray, shapes, s_aabbs, visitor, min_index, remain_prims, cp_base << BASE_SHFL, min_dist);
            __syncthreads();
        }
#endif  // RENDERER_USE_BVH
        // ============= step 2: local shading for indirect bounces ================
        if (min_index >= 0) {
            extract.set_index(min_index);
            auto it = variant::apply_visitor(extract, shapes[min_index]);

            // ============= step 3: next event estimation ================
            // (1) randomly pick one emitter
            int object_id   = prim2obj[min_index],
                material_id = objects[object_id].bsdf_id,
                emitter_id  = objects[object_id].emitter_id;
            hit_emitter = emitter_id > 0;

            // emitter MIS
            emission_weight = emission_weight / (emission_weight + 
                    objects[object_id].solid_angle_pdf(it.shading_norm, ray.d, min_dist) * hit_emitter * (b > 0));

            float direct_pdf = 1;       // direct_pdf is the product of light_sampling_pdf and emitter_pdf
            // (2) check if the ray hits an emitter
            Vec4 direct_comp = throughput *\
                        c_emitter[emitter_id]->eval_le(&ray.d, &it.shading_norm);
            radiance += direct_comp * emission_weight;

            Emitter* emitter = sample_emitter(sampler, direct_pdf, num_emitter, emitter_id);
            // (3) sample a point on the emitter (we avoid sampling the hit emitter)
            emitter_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(sampler.discrete1D(), direct_pdf);
            Ray shadow_ray(ray.advance(min_dist), Vec3(0, 0, 0));
            // use ray.o to avoid creating another shadow_int variable
            shadow_ray.d = emitter->sample(shadow_ray.o, direct_comp, direct_pdf, sampler.next2D(), verts, norms, emitter_id) - shadow_ray.o;
            
            float emit_len_mis = shadow_ray.d.length();
            shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direction

            // (3) NEE scene intersection test (possible warp divergence, but... nevermind)
            if (emitter != c_emitter[0] &&
#ifdef RENDERER_USE_BVH
            occlusion_test_bvh(shadow_ray, shapes, bvh_fronts, bvh_backs, node_fronts, 
                        node_backs, node_offsets, *verts, node_num, emit_len_mis - EPSILON)
#else   // RENDERER_USE_BVH
            occlusion_test(shadow_ray, objects, shapes, aabbs, *verts, num_objects, emit_len_mis - EPSILON)
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
            ray.o = std::move(shadow_ray.o);
            ray.d = c_material[material_id]->sample_dir(ray.d, it, throughput, emission_weight, sampler.next2D());

            if (radiance.numeric_err())
                radiance.fill(0);
        }
    }
    __syncthreads();
    if constexpr (render_once) {
        // image will be the output buffer, there will be double buffering
        auto local_v = image(px, py) + radiance;
        image(px, py) = local_v;
        local_v *= 1.f / float(accum_cnt);
        FLOAT4(output_buffer[(px + py * image.w()) << 2]) = float4(local_v); 
    } else {
        image(px, py) += radiance;
    }
}

template CPT_KERNEL void render_pt_kernel<true>(
    const DeviceCamera& dev_cam, 
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
    DeviceImage& image,
    float* output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int node_num,
    int accum_cnt
);

template CPT_KERNEL void render_pt_kernel<false>(
    const DeviceCamera& dev_cam, 
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
    DeviceImage& image,
    float* output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int node_num,
    int accum_cnt
);