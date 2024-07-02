/**
 * (W)ave(f)ront (S)imple path tracing with stream multiprocessing
 * We first consider how to make a WF tracer, then we start optimizing it, hence this is a 'Simple' one 
 * 
 * for each stream, we will create their own ray pools for
 * stream compaction and possible execution reordering
 * 
 * each stream contains 4 * 4 blocks, each block contains 16 * 16 threads, which is therefore
 * a 64 * 64 pixel patch. We will only create at most 8 streams, to fill up the host-device connections
 * therefore, it is recommended that the image sizes are the multiple of 64
 * 
 * for each kernel function, sx (int) and sy (int) are given, which is the base location of the current
 * stream. For example, let there be 4 streams and 4 kernel calls and the image is of size (256, 256)
 * stream 1: (0, 0), (64, 0), (128, 0), (192, 0)                |  1   2   3   4  |
 * stream 2: (0, 64), (64, 64), (128, 64), (192, 64)            |  1   2   3   4  |
 * stream 3: (0, 128), (64, 128), (128, 128), (192, 128)        |  1   2   3   4  |
 * stream 4: (0, 192), (64, 192), (128, 192), (192, 192)        |  1   2   3   4  |
 * 
 * @author Qianyue He
 * @date   2024.6.20
*/
#pragma once
#include <cuda/pipeline>
#include "core/progress.h"
#include "core/emitter.cuh"
#include "core/object.cuh"
#include "core/stats.h"
#include "renderer/path_tracer.cuh"

struct PathPayLoad {
    Vec4 thp;           // 4 * 4 Bytes
    Vec4 L;             // 4 * 4 Bytes

    Ray ray;            // 8 * 4 Bytes
    Sampler sg;         // 6 * 4 Bytes
    int path_flag;      // 1 * 4 Byte:

    Interaction it;     // 5 * 4 Bytes

    // 28 * 4 Bytes in total
    CONDITION_TEMPLATE_2(T1, T2, Vec3)
    CPT_CPU_GPU PathPayLoad(T1&& o_, T2&& d_, int seed, float hitT = MAX_DIST, int offset = 0):
        thp(1, 1, 1, 1), L(0, 0, 0, 1),
        ray(std::forward<T1>(o_), std::forward<T2>(d_), hitT), sg(seed, offset) {}

    CPT_CPU_GPU PathPayLoad(float vthp, float vl, int seed, int offset = 0):
        thp(vthp, vthp, vthp), L(vl, vl, vl),
        ray(Vec3(0, 0, 0), Vec3(0, 0, 1), MAX_DIST), sg(seed, offset) {}

    CPT_CPU_GPU_INLINE void reset() {
        ray.clr_hit();
        ray.set_hit_index(0);
    }
};

namespace {
    using PayLoadBuffer = PathPayLoad* const;
    using ConstPayLoadBuffer = const PayLoadBuffer;
}

// camera is defined in the global constant memory
// extern __constant__ DeviceCamera dev_cam;

/**
 * @brief ray generation kernel 
 * @param ray_pool the writing destination for the raygen result
 * @param samplers samplers to be used to generate random number
 * @param (sx, sy) stream offset for the current image patch
 * 
 * @note we first consider images that have width and height to be the multiple of 128
 * to avoid having to consider the border problem
*/ 
CPT_KERNEL void raygen_shader(PayLoadBuffer payloads, int* const idx_buffer, int sx, int sy) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x + sx, py = threadIdx.y + blockIdx.y * blockDim.y + sy;
    const int block_index = (py - sy) * blockDim.x * gridDim.x + px - sx;
    payloads[block_index].ray = dev_cam.generate_ray(px, py, payloads[block_index].sg.next2D());
    idx_buffer[block_index] = block_index;
    __syncthreads();
}

/**
 * @brief find ray intersection for next hit pos
 * We first start with small pool size (4096), which can comprise at most 16 blocks
 * The ray pool is stream-compacted (with thrust::parition to remove the finished)
 * Note that we need an index buffer, since the Ray and Sampler are coupled
 * and we need the index to port the 
*/ 
CPT_KERNEL void closesthit_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    PayLoadBuffer payloads,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    const int* const idx_buffer,
    int num_prims,
    int num_valid
) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px, tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    if (block_index < num_valid) {
        int index = idx_buffer[block_index];
        PathPayLoad payload = payloads[index];        // To local register
        payload.reset();
        
        __shared__ Vec3 s_verts[TRI_IDX(BASE_ADDR)];         // vertex info
        __shared__ AABBWrapper s_aabbs[BASE_ADDR];            // aabb

        ArrayType<Vec3> s_verts_arr(reinterpret_cast<Vec3*>(&s_verts[0]), BASE_ADDR);
        ShapeIntersectVisitor visitor(s_verts_arr, payload.ray, 0);
        ShapeExtractVisitor extract(*verts, *norms, *uvs, payload.ray, 0);

        Vec4 throughput(1, 1, 1), radiance(0, 0, 0);

        int num_copy = (num_prims + BASE_ADDR - 1) / BASE_ADDR, min_index = -1;   // round up
        payload.ray.hit_t = MAX_DIST;

        // ============= step 1: ray intersection =================
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
            payload.ray.hit_t = ray_intersect(payload.ray, shapes, s_aabbs, visitor, min_index, remain_prims, cp_base << BASE_SHFL, payload.ray.hit_t);
            __syncthreads();
        }

        // ============= step 2: local shading for indirect bounces ================
        if (min_index >= 0 && payload.thp.good()) {
            extract.set_index(min_index);
            payload.ray.set_hit();
            payload.ray.set_hit_index(min_index);
            payload.it = variant::apply_visitor(extract, shapes[min_index]);
        }
        if (!payload.thp.good()) {
            payload.reset();
            payload.ray.set_active(false);
        }

        payloads[index].ray = payload.ray;
        payloads[index].sg  = payload.sg;
        payloads[index].it  = payload.it;
    }
    __syncthreads();
}

/***
 * For non-delta hit (shading point), direct component should be evaluated:
 * we sample a light source then start ray intersection test
*/
CPT_KERNEL void nee_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    PayLoadBuffer payloads,
    ConstPrimPtr norms, 
    ConstUVPtr,         
    const int* const idx_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int num_valid
) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px, tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    if (block_index < num_valid) {
        int index = idx_buffer[block_index];
        PathPayLoad payload = payloads[index];        // To local register

        int object_id   = prim2obj[payload.ray.hit_id()],
            material_id = objects[object_id].bsdf_id,
            emitter_id  = objects[object_id].emitter_id;
        bool hit_emitter = emitter_id > 0;

        float direct_pdf = 1;

        Emitter* emitter = sample_emitter(payload.sg, direct_pdf, num_emitter, emitter_id);
        int emitter_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(payload.sg.discrete1D(), direct_pdf);
        Ray shadow_ray(payload.ray.advance(payload.ray.hit_t), Vec3(0, 0, 0));
        // use ray.o to avoid creating another shadow_int variable
        Vec4 direct_comp(0, 0, 0, 1);
        shadow_ray.d = emitter->sample(shadow_ray.o, direct_comp, direct_pdf, payload.sg.next2D(), verts, norms, emitter_id) - shadow_ray.o;

        float emit_len_mis = shadow_ray.d.length();
        shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized directi 
        // (3) NEE scene intersection test (possible warp divergence, but... nevermind)
        if (emitter != c_emitter[0] && occlusion_test(shadow_ray, objects, shapes, aabbs, *verts, num_objects, emit_len_mis - EPSILON)) {
            // MIS for BSDF / light sampling, to achieve better rendering
            // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
            emit_len_mis = direct_pdf + c_material[material_id]->pdf(payload.it, shadow_ray.d, payload.ray.d) * emitter->non_delta();
            payload.L += payload.thp * direct_comp * c_material[material_id]->eval(payload.it, shadow_ray.d, payload.ray.d) * \
                (float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
            // numerical guard, in case emit_len_mis is 0
        }

        payloads[index].L   = payloads->L;
        payloads[index].thp = payloads->thp;
        payloads[index].sg  = payloads->sg;
    }
    __syncthreads();
}


/**
 * BSDF sampling & direct shading shader
*/

CPT_KERNEL void bsdf_emission_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    PayLoadBuffer payloads,
    ConstUVPtr,         
    const int* const idx_buffer,
    int num_prims, 
    int num_valid,
    bool secondary_bounce
) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px;

    if (block_index < num_valid) {
        int index = idx_buffer[block_index];
        PathPayLoad payload = payloads[index];        // To local register

        int object_id   = prim2obj[payload.ray.hit_id()],
            material_id = objects[object_id].bsdf_id,
            emitter_id  = objects[object_id].emitter_id;
        bool hit_emitter = emitter_id > 0;

        // emitter MIS
        float emission_weight = emission_weight / (emission_weight + 
                objects[object_id].solid_angle_pdf(payload.it.shading_norm, payload.ray.d, payload.ray.hit_t) * hit_emitter * secondary_bounce);
        // (2) check if the ray hits an emitter
        Vec4 direct_comp = payload.thp *\
                    c_emitter[emitter_id]->eval_le(&payload.ray.d, &payload.it.shading_norm);
        payload.L += direct_comp * emission_weight;

        payload.ray.o = payload.ray.advance(payload.ray.hit_t);
        payload.ray.d = c_material[material_id]->sample_dir(payload.ray.d, payload.it, payload.thp, emission_weight, payload.sg.next2D());

        payloads[index].ray = payload.ray;
        payloads[index].thp = payload.thp;
        payloads[index].sg  = payload.sg;
        payloads[index].L   = payload.L;
    }
    __syncthreads();
}

/**
 * Sample the new ray according to BSDF
*/
CPT_KERNEL void ray_update_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    PayLoadBuffer payloads,
    ConstUVPtr,   
    const int* const idx_buffer,
    int num_valid
) {

    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px;

    if (block_index < num_valid) {
        int index = idx_buffer[block_index];
        PathPayLoad payload = payloads[index];        // To local register

        int object_id   = prim2obj[payload.ray.hit_id()],
            material_id = objects[object_id].bsdf_id,
            emitter_id  = objects[object_id].emitter_id;
        
        PathPayLoad payload = payloads[block_index];
        payload.ray.o = payload.ray.advance(payload.ray.hit_t);
        // TODO: fix this path PDF
        payload.ray.d = c_material[material_id]->sample_dir(
            payload.ray.d, payload.it, payload.thp, emission_weight, payload.sg.next2D()
        );
        
    }
    __syncthreads();
}


CPT_KERNEL void miss_shader(
    PayLoadBuffer payloads,
    const int* const idx_buffer,
    int num_valid
) {
    // Nothing here, currently, if we decide not to support env lighting

    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px;

    if (block_index < num_valid) {
        int index = idx_buffer[block_index];
        payloads[index].reset();
        payloads[index].ray.set_active(false);
    }
    __syncthreads();
}

CPT_KERNEL void radiance_splat(
    PayLoadBuffer payloads,
    DeviceImage image,
    int* const idx_buffer,
    int num_valid
) {
    // Nothing here, currently, if we decide not to support env lighting

    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px;
    Vec4 L = payloads[block_index].L;
    image(px, py) += L.numeric_err() ? Vec4(0, 0, 0, 1) : L;
    __syncthreads();
}
