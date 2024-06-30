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

#include "renderer/tracer_base.cuh"

namespace {
    using RayPool = Ray*;
    using ConstRayPool = const RayPool const;
}

struct PathPayLoad {
    Vec4 thp;
    Vec4 L;

    CPT_CPU_GPU PathPayLoad() {}
    CPT_CPU_GPU PathPayLoad(float thp_v = 1, float l_v = 0):
        thp(thp_v, thp_v, thp_v, 1), L(l_v, l_v, l_v, 1) {}
};

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
CPT_KERNEL void raygen_shader(RayPool ray_pool, Sampler* const samplers, PathPayLoad* payloads, int sx, int sy) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x + sx, py = threadIdx.y + blockIdx.y * blockDim.y + sy;
    const int block_index = py * blockDim.x * gridDim.x + px;
    ray_pool[block_index] = dev_cam.generate_ray(px, py, samplers[block_index].next2D());
}

// 

/**
 * @brief find ray intersection for next hit pos
 * We first start with small pool size (4096), which can comprise at most 16 blocks
 * The ray pool is stream-compacted (with thrust::parition to remove the finished)
 * Note that we need an index buffer, since the Ray and Sampler are coupled
 * and we need the index to port the 
 * 
 * @param ray_pool the writing destination for the raygen result
 * @param samplers samplers to be used to generate random number
 * @param (sx, sy) stream offset for the current image patch
 * 
 * @note we first consider images that have width and height to be the multiple of 64
 * to avoid having to consider the border problem
*/ 
CPT_KERNEL void closesthit_shader(RayPool ray_pool, Sampler* const samplers, PathPayLoad* payloads) {
    
}