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

/**
 * @param verts     vertices, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms     normal vectors, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs       uv coordinates, SoA3: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
 * @param camera    GPU camera model (constant memory)
 * @param image     GPU image buffer
 * @param num_prims number of primitives (to be intersected with)
 * @param max_depth maximum allowed bounce
*/

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
    const DeviceCamera& camera,
    DeviceImage& image,
    int num_prims,
    int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
) {
    int px = threadIdx.x + blockIdx.x * gridDim.x, py = threadIdx.y + blockIdx.y * gridDim.y;

    Sampler sampler(px + py * image.w());
    // step 1: generate ray
    Ray ray = camera.generate_ray(px, py, sampler);

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
        image(px, py) += min_dist;
    }
}

CPT_CPU std::vector<uint8_t> render_depth(
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    const DeviceCamera& camera,
    int num_prims,
    int width  = 800,
    int height = 800,
    int num_iter  = 64,
    int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
) {
    DeviceImage image, *dev_image;
    CUDA_CHECK_RETURN(cudaMalloc(&dev_image, sizeof(DeviceImage)));
    CUDA_CHECK_RETURN(cudaMemcpy(dev_image, &image, sizeof(DeviceImage), cudaMemcpyHostToDevice));

    for (int i = 0; i < num_iter; i++) {
        // for more sophisticated renderer (like path tracer), shared_memory should be used
        render_depth_kernel<<<dim3(width >> 4, height >> 4), dim3(16, 16)>>>(
            shapes, aabbs, verts, norms, uvs, camera, *dev_image, num_prims, max_depth); 
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    return image.export_cpu();
}