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
    Sampler sp;         // 6 * 4 Bytes
    uint32_t _pad;      // 1 * 4 Byte

    Interaction it;     // 5 * 4 Bytes

    // 28 * 4 Bytes in total
    CONDITION_TEMPLATE_2(T1, T2, Vec3)
    CPT_CPU_GPU PathPayLoad(T1&& o_, T2&& d_, int seed, float hitT = MAX_DIST, int offset = 0):
        thp(1, 1, 1, 1), L(0, 0, 0, 1),
        ray(std::forward<T1>(o_), std::forward<T2>(d_), hitT), sp(seed, offset) {}

    CPT_CPU_GPU PathPayLoad(float vthp, float vl, int seed, int offset = 0):
        thp(vthp, vthp, vthp), L(vl, vl, vl),
        ray(Vec3(0, 0, 0), Vec3(0, 0, 1), MAX_DIST), sp(seed, offset) {}
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
CPT_KERNEL void raygen_shader(PayLoadBuffer payloads, int sx, int sy) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x + sx, py = threadIdx.y + blockIdx.y * blockDim.y + sy;
    const int block_index = (py - sy) * blockDim.x * gridDim.x + px - sx;
    payloads[block_index].ray = dev_cam.generate_ray(px, py, payloads[block_index].sp.next2D());
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
    int num_prims,
    int* const idx_buffer,
    int num_valid
) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px, tid = threadIdx.x + threadIdx.y * blockDim.x;
    // block 
    
    if (block_index < num_valid) {
        int index = idx_buffer[block_index];
        PathPayLoad payload = payloads[index];        // To local register
        
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
        payload.ray.set_hit_status(min_index >= 0);
        payload.ray.set_hit_index(min_index >= 0 ? min_index : 0);
        if (min_index >= 0) {
            extract.set_index(min_index);
            payload.it = variant::apply_visitor(extract, shapes[min_index]);
        }

        payloads[index] = payload;
    }
}