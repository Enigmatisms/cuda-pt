/**
 * Wavefront path tracing with stream multiprocessing
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

#include "renderer/tracer_base.cuh"

namespace {
    using RayPool = Ray*;
    using ConstRayPool = const RayPool const;
}

// camera is defined in the global constant memory
// extern __constant__ DeviceCamera dev_cam;

// ray generation kernel 
CPT_KERNEL void raygen_shader(RayPool ray_pool, CudaSampler* const samplers, int sx, int sy) {
    int px = threadIdx.x + blockIdx.x * blockDim.x + sx, py = threadIdx.y + blockIdx.y * blockDim.y + sy;
    // Ray ray = dev_cam.generate_ray(px, py, sampler);
}