/**
 * Simple tile-based depth renderer
 * @date: 5.6.2024
 * @author: Qianyue He
*/

#include <variant/variant.h>
#include "core/soa.cuh"
#include "core/camera_model.cuh"
#include "core/host_device.cuh"
#include "core/shapes.cuh"

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

__global__ void render(

    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    const DeviceCamera& camera,
    DeviceImage& image,
    int num_prims,
    int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
) {
    int tile_x = blockIdx.x, tile_y = blockIdx.y;
    int px = threadIdx.x + tile_x * gridDim.x, py = threadIdx.y + tile_y * gridDim.y;

    Sampler sampler(px + py * image.w());
    // step 1: generate ray
    Ray ray = camera.generate_ray(px, py, sampler);

    // step 2: bouncing around the scene until the max depth is reached
    for (int b = 0; b < max_depth; b++) {
        // step 3: traverse all the primitives and intersect

    }

}