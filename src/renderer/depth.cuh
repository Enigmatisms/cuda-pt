/**
 * Simple tile-based depth renderer
 * @date: 5.6.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include "core/stats.h"
#include "renderer/tracer_base.cuh"

/**
 * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
 * @param camera    GPU camera model (constant memory)
 * @param image     GPU image buffer
 * @param num_prims number of primitives (to be intersected with)
 * @param max_depth maximum allowed bounce
*/
CPT_KERNEL static void render_depth_kernel(
    const DeviceCamera& dev_cam,
    const PrecomputedArray& verts,
    ConstAABBPtr aabbs,
    ConstNormPtr norms, 
    ConstUVPtr uvs,
    DeviceImage image,
    int num_prims,
    int max_bounce = 1/* max depth, useless for depth renderer, 1 anyway */
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    Sampler sampler(px + py * image.w());
    // step 1: generate ray
    Ray ray = dev_cam.generate_ray(px, py, sampler);

    // step 2: bouncing around the scene until the max depth is reached
    Interaction it;

    // A matter of design choice
    // optimization: copy at most 32 prims from global memory to shared memory

    __shared__ Vec4 s_verts[TRI_IDX(32)];         // vertex info
    __shared__ AABBWrapper s_aabbs[32];            // aabb
    PrecomputedArray s_verts_arr(reinterpret_cast<Vec4*>(&s_verts[0]), BASE_ADDR);

    int num_copy = (num_prims + 31) / 32;   // round up
    float min_dist = MAX_DIST;
    for (int b = 0; b < max_bounce; b++) {
        for (int cp_base = 0; cp_base < num_copy; ++cp_base) {
            // memory copy to shared memory
            int cp_base_5 = cp_base << 5, cur_idx = cp_base_5 + tid, remain_prims = min(num_prims - cp_base_5, 32);
            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
            if (tid < 32 && cur_idx < num_prims) {        // copy from gmem to smem
                cuda::memcpy_async(&s_verts[TRI_IDX(tid)], &verts.data[TRI_IDX(cur_idx)], sizeof(Vec4) * 3, pipe);
                s_aabbs[tid].aabb.copy_from(aabbs[cur_idx]);
            }
            pipe.producer_commit();
            pipe.consumer_wait();
            __syncthreads();
            // step 3: traverse all the primitives and intersect

            int min_index = 0, object_id = 0;
            float prim_u = 0, prim_v = 0;
            min_dist = ray_intersect(s_verts_arr, ray, s_aabbs, remain_prims, 
                cp_base << BASE_SHFL, min_index, object_id, prim_u, prim_v, min_dist);
            __syncthreads();
        }
    }
    image(px, py) += min_dist * (min_dist < MAX_DIST);
}

class DepthTracer: public TracerBase {
using TracerBase::aabbs;
using TracerBase::verts;
using TracerBase::norms; 
using TracerBase::uvs;
using TracerBase::image;
using TracerBase::num_prims;
using TracerBase::w;
using TracerBase::h;

DeviceCamera* camera;
public:
    /**
     * @param shapes    shape information (for AABB generation)
     * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
    */
    DepthTracer(
        const std::vector<Shape>& _shapes,
        const PrecomputedArray& _verts,
        const ArrayType<Vec3>& _norms, 
        const ArrayType<Vec2>& _uvs,
        const DeviceCamera& cam,
        int width, int height
    ): TracerBase(_shapes, _verts, _norms, _uvs, width, height) {
        CUDA_CHECK_RETURN(cudaMalloc(&camera, sizeof(DeviceCamera)));
        CUDA_CHECK_RETURN(cudaMemcpy(camera, &cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice));
    }

    ~DepthTracer() {
        CUDA_CHECK_RETURN(cudaFree(camera));
    }

    CPT_CPU std::vector<uint8_t> render(
        int num_iter = 64,
        int max_depth = 1,/* max depth, useless for depth renderer, 1 anyway */
        bool gamma_correction = true
    ) override {
        printf("Depth tracing. Num primitives: %d, max_depth: %d\n", num_prims, max_depth);
        TicToc _timer("render_kernel()", num_iter);
        for (int i = 0; i < num_iter; i++) {
            // for more sophisticated renderer (like path tracer), shared_memory should be used
            render_depth_kernel<<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
                    *camera, *verts, aabbs, norms, uvs, image, num_prims, max_depth); 
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }
        return image.export_cpu(1.f / (15.f * num_iter), gamma_correction);
    }


};

