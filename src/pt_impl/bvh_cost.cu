/**
 * @file path_tracer.cu
 * @author Qianyue He
 * @brief Megakernel Path Tracer implementation
 * @date 2024.10.10
 * @copyright Copyright (c) 2024
 */

#include "core/xyz.cuh"
#include "core/progress.h"
#include "core/serialize.h"
#include "core/color_map.cuh"
#include "renderer/bvh_cost.cuh"

static constexpr int SEED_SCALER = 11467;           //-4!
static constexpr int SHFL_THREAD_X = 5;             // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16 threads
static constexpr int SHFL_THREAD_Y = 2;             // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16 threads

CPT_GPU int ray_intersect_cost(
    const Ray& ray,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    const PrecomputedArray& verts,
    ConstU4Ptr cached_nodes,
    int& min_index,
    const int node_num,
    const int cache_num
) {
    int node_idx     = 0;
    float aabb_tmin  = 0, min_dist = MAX_DIST;
    int intersect_query = 0;
    // There can be much control flow divergence, not good
    Vec3 inv_d = ray.d.rcp(), o_div = ray.o * inv_d; 
    for (int i = 0; i < cache_num;) {
        const CompactNode node(cached_nodes[i]);
        bool intersect_node = node.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < min_dist;
        int all_offset = node.get_cached_offset(), gmem_index = node.get_gmem_index();
        int increment = (!intersect_node) * all_offset + int(intersect_node && all_offset != 1);
        // reuse
        intersect_node = intersect_node && all_offset == 1;
        i = intersect_node ? cache_num : (i + increment);
        node_idx = intersect_node ? gmem_index : node_idx;
        intersect_query ++;
    }
    // There can be much control flow divergence, not good
    while (node_idx < node_num) {
        const CompactNode node(tex1Dfetch<uint4>(nodes, node_idx));
        int beg_idx = node.get_beg_idx(), end_idx = node.get_prim_cnt();
        bool intersect_node = node.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < min_dist;
        // The logic here: end_idx is reuse, if end_idx < 0, meaning that the current node is
        // non-leaf, non-leaf node stores (-all_offset) as end_idx, so to skip the node and its children
        // -end_idx will be the offset. While for leaf node, 1 will be the increment offset, and `POSITIVE` end_idx
        // is stored. So the following for loop can naturally run (while for non-leaf, naturally skip)
        node_idx += (!intersect_node) * (beg_idx < 0 ? -beg_idx : 1) + int(intersect_node);
        end_idx = intersect_node && beg_idx >= 0 ? end_idx + beg_idx : beg_idx;
        intersect_query ++;
        for (int idx = beg_idx; idx < end_idx; idx ++) {
            // if current ray intersects primitive at [idx], tasks will store it
            int obj_idx = tex1Dfetch<int>(bvh_leaves, idx);
#ifdef TRIANGLE_ONLY
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v);
#else
            float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, verts, idx, it_u, it_v, obj_idx >= 0);
#endif
            bool valid = dist > EPSILON && dist < min_dist;
            min_dist = valid ? dist : min_dist;
            min_index = valid ? idx : min_index;
            intersect_query ++;
        }
    }
    return intersect_query;
}

CPT_KERNEL static void render_cost_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstU4Ptr cached_nodes,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int2* __restrict__ global_min_max,
    int seed_offset,
    int node_num,
    int accum_cnt,
    int cache_num
) {
    extern __shared__ uint4 s_cached[];
    __shared__ int2 min_max;

    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    Sampler sampler(px + py * image.w(), seed_offset);

    int min_index = -1, tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid == 0) min_max = make_int2(10000, 0);
    // cache near root level BVH nodes for faster traversal
    if (tid < cache_num) {      // no more than 32 nodes will be cached
        s_cached[tid] = cached_nodes[tid];
    }
    Ray ray = dev_cam.generate_ray(px, py, sampler);
    __syncthreads();
    
    // ============= step 1: ray intersection =================
    int intrsct_num = ray_intersect_cost(
        ray, bvh_leaves, nodes, verts, s_cached, 
        min_index, node_num, cache_num
    );
    
    // ============= step 2: local shading for indirect bounces ================
    // image will be the output buffer, there will be double buffering
    if (min_index >= 0) {
        auto local_v = image(px, py) + float(intrsct_num);
        image(px, py) = local_v;
        atomicMin(&min_max.x, intrsct_num);
        atomicMax(&min_max.y, intrsct_num);
    }
    __syncthreads();
    if (tid == 0) {
        atomicMin(&global_min_max->x, min_max.x);
        atomicMax(&global_min_max->y, min_max.y);
    }
}

BVHCostVisualizer::~BVHCostVisualizer() {
    color_map_id = -1;
    printf("[Renderer] BVH Cost Visualizer Object destroyed.\n");
}

CPT_CPU void BVHCostVisualizer::render_online(
    const MaxDepthParams& md,
    bool gamma_corr
) {
    min_max->x = 10000;
    min_max->y = 0;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0, cached_size = std::max(num_cache * sizeof(uint4), sizeof(uint4));
    // if we have an illegal memory access here: check whether you have a valid emitter in the xml scene description file.
    // it might be possible that having no valid emitter triggers an illegal memory access
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));
    accum_cnt ++;
    render_cost_kernel<<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
        *camera, verts, bvh_leaves, nodes, _cached_nodes, image, output_buffer,
        min_max, accum_cnt * SEED_SCALER, num_nodes, accum_cnt, num_cache
    ); 
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    false_color_mapping<<<dim3(w >> 5, h >> 3), dim3(32, 8)>>>(image, output_buffer, color_map_id, accum_cnt, *min_max, false);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}