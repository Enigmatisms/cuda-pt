/**
 * Simple tile-based depth renderer
 * @date: 5.6.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/stats.h"
#include "renderer/tracer_base.cuh"

extern __constant__ DeviceCamera dev_cam;

/**
 * @param verts     vertices, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms     normal vectors, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs       uv coordinates, SoA3: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
 * @param camera    GPU camera model (constant memory)
 * @param image     GPU image buffer
 * @param num_prims number of primitives (to be intersected with)
 * @param max_depth maximum allowed bounce
*/
__global__ static void render_depth_kernel(
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    DeviceImage& image,
    int num_prims,
    int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
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

    __shared__ Vec3 s_verts[3][32];         // vertex info
    __shared__ AABB s_aabbs[32];            // aabb
    ShapeIntersectVisitor visitor(*verts, ray, 0);

    int num_copy = (num_prims + 31) / 32;   // round up
    float min_dist = MAX_DIST;
    for (int b = 0; b < max_depth; b++) {
        for (int cp_base = 0; cp_base < num_copy; ++cp_base) {
            // memory copy to shared memory
            int cp_base_5 = cp_base << 5, cur_idx = cp_base_5 + tid, remain_prims = min(num_prims - cp_base_5, 32);
            if (tid < 32 && cur_idx < num_prims) {        // copy from gmem to smem
                s_verts[0][tid] = verts->x[cur_idx];
                s_verts[1][tid] = verts->y[cur_idx];
                s_verts[2][tid] = verts->z[cur_idx];
                s_aabbs[tid]    = aabbs[cur_idx];
            }
            __syncthreads();
            // step 3: traverse all the primitives and intersect

            int min_index = 0;
            min_dist = ray_intersect(ray, shapes, s_aabbs, visitor, min_index, remain_prims, cp_base_5, min_dist);
        }
    }
    image(px, py) += min_dist * (min_dist < MAX_DIST);
}

class DepthTracer: public TracerBase {
using TracerBase::shapes;
using TracerBase::aabbs;
using TracerBase::verts;
using TracerBase::norms; 
using TracerBase::uvs;
using TracerBase::image;
using TracerBase::dev_image;
using TracerBase::num_prims;
using TracerBase::w;
using TracerBase::h;
public:
    /**
     * @param shapes    shape information (for ray intersection)
     * @param verts     vertices, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param uvs       uv coordinates, SoA3: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
    */
    DepthTracer(
        const std::vector<Shape>& _shapes,
        const SoA3<Vec3>& _verts,
        const SoA3<Vec3>& _norms, 
        const SoA3<Vec2>& _uvs,
        int width, int height
    ): TracerBase(_shapes, _verts, _norms, _uvs, width, height) {}

    CPT_CPU std::vector<uint8_t> render(
        int num_iter = 64,
        int max_depth = 1,/* max depth, useless for depth renderer, 1 anyway */
        bool gamma_correction = true
    ) override {
        TicToc _timer("render_kernel()", num_iter);
        for (int i = 0; i < num_iter; i++) {
            // for more sophisticated renderer (like path tracer), shared_memory should be used
            render_depth_kernel<<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
                    shapes, aabbs, verts, norms, uvs, *dev_image, num_prims, max_depth); 
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }
        return image.export_cpu(1.f / (5.f * num_iter), gamma_correction);
    }
};

