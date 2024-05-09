/**
 * Simple tile-based depth renderer
 * @date: 5.6.2024
 * @author: Qianyue He
*/

#include <variant/variant.h>
#include "core/soa.cuh"
#include "core/shapes.cuh"
#include "core/camera_model.cuh"
#include "core/host_device.cuh"
#include "core/stats.h"

/**
 * @param verts     vertices, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms     normal vectors, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs       uv coordinates, SoA3: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
 * @param camera    GPU camera model (constant memory)
 * @param image     GPU image buffer
 * @param num_prims number of primitives (to be intersected with)
 * @param max_depth maximum allowed bounce
*/

extern __constant__ DeviceCamera dev_cam;

namespace {
    using Shape = variant::variant<SphereShape, TriangleShape>;
    using ConstShapePtr = const Shape* const;
}

static constexpr float MAX_DIST = 1e7;

__global__ void render_depth_kernel(
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

    Sampler sampler(px + py * image.w());
    // step 1: generate ray
    Ray ray = dev_cam.generate_ray(px, py, sampler);

    // step 2: bouncing around the scene until the max depth is reached
    Interaction it;
    ShapeVisitor visitor(verts, norms, uvs, &ray, &it, 0);
    for (int b = 0; b < max_depth; b++) {
        // step 3: traverse all the primitives and intersect
        float min_dist = MAX_DIST, aabb_tmin = 0;
        for (int idx = 0; idx < num_prims; idx ++) {
            // (1) AABB test, if failed then the thread will be idle
            if (aabbs[idx].intersect(ray, aabb_tmin) && aabb_tmin <= min_dist) {
                visitor.set_index(idx);
                float dist = variant::apply_visitor(visitor, shapes[idx]);
                bool valid = dist > epsilon;
                min_dist = min(min_dist, dist) * valid + min_dist * (1 - valid);
            }
            __syncwarp();           // where to place this syncwarp function?
        }
        image(px, py) += min_dist * (min_dist < MAX_DIST);
    }
}

CPT_CPU std::vector<uint8_t> render_depth(
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    const SoA3<Vec3>& verts,
    const SoA3<Vec3>& norms, 
    const SoA3<Vec2>& uvs,
    int num_prims,
    int width  = 800,
    int height = 800,
    int num_iter  = 64,
    int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
) {
    ProfilePhase _(Prof::DepthRenderingHost);
    DeviceImage image(width, height);
    auto dev_image = to_gpu(image);
    auto verts_dev = to_gpu(verts);
    auto norms_dev = to_gpu(norms);
    auto uvs_dev   = to_gpu(uvs);

    {
        ProfilePhase _p(Prof::DepthRenderingDevice);
        for (int i = 0; i < num_iter; i++) {
            // for more sophisticated renderer (like path tracer), shared_memory should be used
            render_depth_kernel<<<dim3(width >> 4, height >> 4), dim3(16, 16)>>>(
                shapes, aabbs, verts_dev, norms_dev, uvs_dev, *dev_image, num_prims, max_depth); 
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }
    }
    
    CUDA_CHECK_RETURN(cudaFree(dev_image));
    CUDA_CHECK_RETURN(cudaFree(verts_dev));
    CUDA_CHECK_RETURN(cudaFree(norms_dev));
    CUDA_CHECK_RETURN(cudaFree(uvs_dev));
    return image.export_cpu(1.f / (5.f * num_iter));
}