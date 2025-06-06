// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * @author Qianyue He
 * @brief Megakernel Path Tracer implementation
 * @date 2024.10.10
 */

#include "core/color_map.cuh"
#include "core/progress.h"
#include "core/serialize.h"
#include "core/xyz.cuh"
#include "renderer/bvh_cost.cuh"

static constexpr int SEED_SCALER = 11467; //-4!
static constexpr int SHFL_THREAD_X =
    5; // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16
       // threads
static constexpr int SHFL_THREAD_Y =
    2; // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16
       // threads

CPT_GPU Vec4 ray_intersect_cost(const Ray &ray,
                                const cudaTextureObject_t bvh_leaves,
                                const cudaTextureObject_t nodes,
                                const PrecomputedArray &verts,
                                ConstF4Ptr cached_nodes, int &min_index,
                                const int node_num, const int cache_num) {
    int node_idx = 0;
    float aabb_tmin = 0, min_dist = MAX_DIST;
    Vec4 render_cost(0, 0, 0);
    // There can be much control flow divergence, not good
    Vec3 inv_d = ray.d.rcp(), o_div = ray.o * inv_d;
    for (int i = 0; i < cache_num;) {
        const CompactNode node(cached_nodes[i]);
        bool intersect_node =
            node.intersect(inv_d, o_div, aabb_tmin) && aabb_tmin < min_dist;
        int all_offset = node.get_cached_offset(),
            gmem_index = node.get_gmem_index();
        int increment = (!intersect_node) * all_offset +
                        int(intersect_node && all_offset != 1);
        // reuse
        intersect_node = intersect_node && all_offset == 1;
        i = intersect_node ? cache_num : (i + increment);
        node_idx = intersect_node ? gmem_index : node_idx;
        render_cost.x() += 1.f;
    }
    // There can be much control flow divergence, not good
    while (node_idx < node_num) {
        const LinearNode node(tex1Dfetch<float4>(nodes, 2 * node_idx),
                              tex1Dfetch<float4>(nodes, 2 * node_idx + 1));
        int beg_idx = 0, end_idx = 0;
        node.get_range(beg_idx, end_idx);
        bool intersect_node = node.aabb.intersect(inv_d, o_div, aabb_tmin) &&
                              aabb_tmin < min_dist;
        // The logic here: end_idx is reuse, if end_idx < 0, meaning that the
        // current node is non-leaf, non-leaf node stores (-all_offset) as
        // end_idx, so to skip the node and its children -end_idx will be the
        // offset. While for leaf node, 1 will be the increment offset, and
        // `POSITIVE` end_idx is stored. So the following for loop can naturally
        // run (while for non-leaf, naturally skip)
        node_idx += (!intersect_node) * (end_idx < 0 ? -end_idx : 1) +
                    int(intersect_node);
        end_idx = intersect_node ? end_idx + beg_idx : 0;
        render_cost.x() += 1.f;
        for (int idx = beg_idx; idx < end_idx; idx++) {
            // if current ray intersects primitive at [idx], tasks will store it
            int obj_info = tex1Dfetch<int>(bvh_leaves, idx);
#ifdef TRIANGLE_ONLY
            float it_u = 0, it_v = 0,
                  dist = Primitive::intersect(ray, verts, idx, it_u, it_v);
#else
            bool is_triangle = (obj_info & 0x80000000) == 0;
            float it_u = 0, it_v = 0,
                  dist = Primitive::intersect(ray, verts, idx, it_u, it_v,
                                              is_triangle);
#endif
            bool valid = dist > EPSILON && dist < min_dist;
            min_dist = valid ? dist : min_dist;
            min_index = valid ? idx : min_index;
            render_cost.y() += 1.f;
        }
    }
    render_cost.z() = render_cost.x() + render_cost.y();
    return render_cost;
}

CPT_KERNEL static void
render_cost_kernel(const DeviceCamera &dev_cam, const PrecomputedArray verts,
                   const cudaTextureObject_t bvh_leaves,
                   const cudaTextureObject_t nodes, ConstF4Ptr cached_nodes,
                   DeviceImage image, float *__restrict__ output_buffer,
                   int seed_offset, int node_num, int accum_cnt,
                   int cache_num) {
    extern __shared__ uint4 s_cached[];

    int px = threadIdx.x + blockIdx.x * blockDim.x,
        py = threadIdx.y + blockIdx.y * blockDim.y;
    Sampler sampler(px + py * image.w(), seed_offset);

    int min_index = -1, tid = threadIdx.x + threadIdx.y * blockDim.x;
    // cache near root level BVH nodes for faster traversal
    if (tid < cache_num) { // no more than 256 nodes will be cached
        s_cached[tid] = cached_nodes[tid];
        int offset_tid = tid + blockDim.x * blockDim.y;
        if (offset_tid < cache_num)
            s_cached[offset_tid] = cached_nodes[offset_tid];
    }
    Ray ray = dev_cam.generate_ray(px, py, sampler);
    __syncthreads();

    // ============= step 1: ray intersection =================
    Vec4 render_cost =
        ray_intersect_cost(ray, bvh_leaves, nodes, verts, s_cached, min_index,
                           node_num, cache_num);

    // ============= step 2: local shading for indirect bounces ================
    // image will be the output buffer, there will be double buffering
    if (min_index >= 0) {
        auto local_v = image(px, py) + render_cost;
        image(px, py) = local_v;
    }
}

template <int WARP_SIZE = 32> CPT_GPU int warp_reduce_max(int val) {
#pragma unroll
    for (int lane_mask = WARP_SIZE >> 1; lane_mask >= 1; lane_mask >>= 1) {
        val = max(__shfl_xor_sync(0xffffffff, val, lane_mask), val);
    }
    return val;
}

CPT_KERNEL void
false_color_manual_range(DeviceImage image, float *__restrict__ output_buffer,
                         int *__restrict__ reduced_max, int color_map_id,
                         int cost_map_id, const int accum_cnt,
                         const float min_dist, const float max_dist) {
    __shared__ int max_val[8];
    int px = threadIdx.x + blockIdx.x * blockDim.x,
        py = threadIdx.y + blockIdx.y * blockDim.y;
    if (threadIdx.x == 0) {
        max_val[threadIdx.y] = 0;
    }
    float tex_coord = image(px, py)[cost_map_id] / float(accum_cnt);
    __syncthreads();
    int max_v = warp_reduce_max(float_to_ordered_int(tex_coord));
    if (threadIdx.x == 0) {
        max_val[threadIdx.y] = max_v;
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        int to_max = 0;
        if (threadIdx.x < 8)
            to_max = max_val[threadIdx.x];
        to_max = warp_reduce_max(to_max);
        if (threadIdx.x == 0)
            max_val[0] = to_max;
    }

    Vec4 color(0, 1);
    if (tex_coord > 0) {
        tex_coord = (tex_coord - min_dist) / fmaxf(max_dist - min_dist, 1e-4f);
        if (color_map_id & 0x80)
            tex_coord = log2f(tex_coord + 1.f);
        color_map_id &= 0x7f;
        if (color_map_id < 3) {
            color = Vec4(tex1D<float4>(COLOR_MAPS[color_map_id], tex_coord));
        } else {
            color = Vec4(tex_coord, 1);
        }
    }
    FLOAT4(output_buffer[(px + py * image.w()) << 2]) = float4(color);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicMax(reduced_max, max_val[0]);
    }
}

BVHCostVisualizer::BVHCostVisualizer(const Scene &scene)
    : DepthTracer(scene), cost_map_id(0), max_v(0) {
    // We need two extra floats to display the minimum and maximum
    Serializer::push<int>(serialized_data, 0);
    Serializer::push<int>(serialized_data, 0);
    Serializer::push<float>(serialized_data, 1);
    CUDA_CHECK_RETURN(cudaMallocManaged(&reduced_max, sizeof(int)));
}

BVHCostVisualizer::~BVHCostVisualizer() {
    CUDA_CHECK_RETURN(cudaFree(reduced_max));
    color_map_id = -1;
    cost_map_id = 0;
    printf("[Renderer] BVH Cost Visualizer Object destroyed.\n");
}

CPT_CPU void BVHCostVisualizer::param_setter(const std::vector<char> &bytes) {
    color_map_id = Serializer::get<int>(bytes, 0);
    cost_map_id = Serializer::get<int>(bytes, 1);
    max_v = Serializer::get<int>(serialized_data, 2);
}

CPT_CPU void BVHCostVisualizer::render_online(const MaxDepthParams &md,
                                              bool gamma_corr) {
    *reduced_max = 0;

    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0,
           cached_size =
               std::max(2 * num_cache * sizeof(float4), sizeof(float4));
    // if we have an illegal memory access here: check whether you have a valid
    // emitter in the xml scene description file. it might be possible that
    // having no valid emitter triggers an illegal memory access
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer(
        (void **)&output_buffer, &_num_bytes, pbo_resc));
    accum_cnt++;
    render_cost_kernel<<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
                         dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y),
                         cached_size>>>(
        *camera, verts, bvh_leaves, nodes, _cached_nodes, image, output_buffer,
        accum_cnt * SEED_SCALER, num_nodes, accum_cnt, num_cache);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    false_color_manual_range<<<dim3(w >> 5, h >> 3), dim3(32, 8)>>>(
        image, output_buffer, reduced_max, color_map_id, cost_map_id, accum_cnt,
        1.f, max_v);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    int ordered_int =
        (*reduced_max >= 0) ? *reduced_max : *reduced_max ^ 0x7FFFFFFF;
    float max_query = *reinterpret_cast<float *>(&ordered_int);
    if (max_v == 0) {
        max_v = ceilf(max_query);
        Serializer::set<int>(serialized_data, 2, max_v);
    }
    Serializer::set<float>(serialized_data, 3, max_query);
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}

CPT_CPU const float *BVHCostVisualizer::render_raw(const MaxDepthParams &md,
                                                   bool gamma_corr) {
    *reduced_max = 0;
    size_t cached_size =
        std::max(2 * num_cache * sizeof(float4), sizeof(float4));
    // if we have an illegal memory access here: check whether you have a valid
    // emitter in the xml scene description file. it might be possible that
    // having no valid emitter triggers an illegal memory access
    accum_cnt++;
    render_cost_kernel<<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
                         dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y),
                         cached_size>>>(
        *camera, verts, bvh_leaves, nodes, _cached_nodes, image, output_buffer,
        accum_cnt * SEED_SCALER, num_nodes, accum_cnt, num_cache);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    false_color_manual_range<<<dim3(w >> 5, h >> 3), dim3(32, 8)>>>(
        image, output_buffer, reduced_max, color_map_id, cost_map_id, accum_cnt,
        1.f, max_v);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    int ordered_int =
        (*reduced_max >= 0) ? *reduced_max : *reduced_max ^ 0x7FFFFFFF;
    float max_query = *reinterpret_cast<float *>(&ordered_int);
    if (max_v == 0) {
        max_v = ceilf(max_query);
        Serializer::set<int>(serialized_data, 2, max_v);
    }
    Serializer::set<float>(serialized_data, 3, max_query);
    return output_buffer;
}
