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
#include "renderer/depth.cuh"
#include "renderer/megakernel_pt.cuh"

static constexpr int SEED_SCALER = 11467;           //-4!
static constexpr int SHFL_THREAD_X = 5;             // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16 threads
static constexpr int SHFL_THREAD_Y = 2;             // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16 threads
static constexpr int ORDERED_INT_MAX = 0x4b189680;  // = MAX_DIST (1e7)

CPT_GPU_CONST cudaTextureObject_t COLOR_MAPS[3];

CPT_KERNEL static void render_depth_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int2* __restrict__ global_min_max,
    int seed_offset,
    int node_num,
    int accum_cnt,
    int cache_num
) {
    extern __shared__ float4 s_cached[];
    __shared__ int2 min_max;

    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    Sampler sampler(px + py * image.w(), seed_offset);

    int min_index = -1, object_id = 0;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid == 0) min_max = make_int2(ORDERED_INT_MAX, 0);
    // cache near root level BVH nodes for faster traversal
    if (tid < 2 * cache_num) {      // no more than 32 nodes will be cached
        s_cached[tid] = cached_nodes[tid];
    }
    Ray ray = dev_cam.generate_ray(px, py, sampler);
    __syncthreads();
    
    float prim_u = 0, prim_v = 0, min_dist = MAX_DIST;
    // ============= step 1: ray intersection =================
    min_dist = ray_intersect_bvh(
        ray, bvh_leaves, nodes, s_cached, 
        verts, min_index, object_id, 
        prim_u, prim_v, node_num, cache_num, MAX_DIST
    );
    
    // ============= step 2: local shading for indirect bounces ================
    // image will be the output buffer, there will be double buffering
    if (min_index >= 0) {
        auto local_v = image(px, py) + min_dist;
        int ordered_int = float_to_ordered_int(min_dist);
        image(px, py) = local_v;
        atomicMin(&min_max.x, ordered_int);
        atomicMax(&min_max.y, ordered_int);
    }
    __syncthreads();
    if (tid == 0) {
        atomicMin(&global_min_max->x, min_max.x);
        atomicMax(&global_min_max->y, min_max.y);
    }
}

CPT_KERNEL static void false_color_mapping(
    DeviceImage image, 
    float* __restrict__ output_buffer,
    int color_map_id,
    const int accum_cnt,
    const int2 min_max
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    float tex_coord = image(px, py).x() / float(accum_cnt);
    float min_dist  = ordered_int_to_float(min_max.x);
    float max_dist  = ordered_int_to_float(min_max.y);
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
}

DepthTracer::DepthTracer(
    const Scene& scene
): TracerBase(scene), num_nodes(-1), color_map_id(1)
{
    if (scene.bvh_available()) {
        size_t num_bvh  = scene.obj_idxs.size();
        // Comment in case I forget: scene.nodes combines nodes_front and nodes_back
        // So the size of nodes is exactly twice the number of nodes 
        num_nodes = scene.nodes.size() >> 1;
        num_cache = scene.cache_fronts.size();
        CUDA_CHECK_RETURN(cudaMalloc(&_obj_idxs,  num_bvh * sizeof(int)));
        CUDA_CHECK_RETURN(cudaMalloc(&_nodes, 2 * num_nodes * sizeof(float4)));
        CUDA_CHECK_RETURN(cudaMalloc(&_cached_nodes, 2 * num_cache * sizeof(float4)));
        // note that BVH leaf node only stores the primitive to object mapping
        bvh_leaves = createTexture1D<int>(scene.obj_idxs.data(), num_bvh, _obj_idxs);
        nodes      = createTexture1D<float4>(scene.nodes.data(), 2 * num_nodes, _nodes);
        CUDA_CHECK_RETURN(cudaMemcpy(_cached_nodes, scene.cache_fronts.data(), sizeof(float4) * num_cache, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(&_cached_nodes[num_cache], scene.cache_backs.data(), sizeof(float4) * num_cache, cudaMemcpyHostToDevice));
    } else {
        throw std::runtime_error("BVH not available in scene. Abort.");
    }
    CUDA_CHECK_RETURN(cudaMalloc(&camera, sizeof(DeviceCamera)));
    CUDA_CHECK_RETURN(cudaMemcpy(camera, &scene.cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMallocManaged(&min_max, sizeof(int2)));
    create_color_map_texture();
    min_max->x = ORDERED_INT_MAX;
    min_max->y = 0;
    // We need one extra int
    Serializer::push<int>(serialized_data, 1);
}

void DepthTracer::create_color_map_texture() {
    std::array<std::vector<float4>, 3> host_data;
    for (int cmap_i = 0; cmap_i < 3; cmap_i ++) {
        auto& color_map = host_data[cmap_i];
        color_map.reserve(256);
        if (cmap_i == 0) {
            for (int i = 0; i < 256; i++) {
                color_map.emplace_back(make_float4(JET_R[i], JET_G[i], JET_B[i], 1));
            }
        } else if (cmap_i == 1) {
            for (int i = 0; i < 256; i++) {
                color_map.emplace_back(make_float4(PLASMA_R[i], PLASMA_G[i], PLASMA_B[i], 1));
            }
        } else {
            for (int i = 0; i < 256; i++) {
                color_map.emplace_back(make_float4(VIRIDIS_R[i], VIRIDIS_G[i], VIRIDIS_B[i], 1));
            }
        }
        colormaps[cmap_i] = createArrayTexture1D<float4>(color_map.data(), _colormap_data[cmap_i], 256);
    }
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(COLOR_MAPS, &colormaps[0], sizeof(colormaps)));
}

DepthTracer::~DepthTracer() {
    CUDA_CHECK_RETURN(cudaFree(camera));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(bvh_leaves));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(nodes));
    CUDA_CHECK_RETURN(cudaFree(_nodes));
    CUDA_CHECK_RETURN(cudaFree(_obj_idxs));
    CUDA_CHECK_RETURN(cudaFree(_cached_nodes));

    for (int i = 0; i < 3; i++) {
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(colormaps[i]));
        CUDA_CHECK_RETURN(cudaFreeArray(_colormap_data[i]));
    }
    if (color_map_id >= 0)
        printf("[Renderer] Depth Tracer Object destroyed.\n");
}

CPT_CPU std::vector<uint8_t> DepthTracer::render(
    const MaxDepthParams& md,
    int num_iter,
    bool gamma_correction
) {
    std::cerr << "Sorry, offline depth tracer is not implemented.\n";
    throw std::runtime_error("Offline Depth Tracer not implemented.");
}

CPT_CPU void DepthTracer::render_online(
    const MaxDepthParams& md,
    bool gamma_corr
) {
    min_max->x = ORDERED_INT_MAX;
    min_max->y = 0;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0, cached_size = std::max(2 * num_cache * sizeof(float4), sizeof(float4));
    // if we have an illegal memory access here: check whether you have a valid emitter in the xml scene description file.
    // it might be possible that having no valid emitter triggers an illegal memory access
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));
    accum_cnt ++;
    render_depth_kernel<<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
        *camera, verts, bvh_leaves, nodes, _cached_nodes, image, output_buffer,
        min_max, accum_cnt * SEED_SCALER, num_nodes, accum_cnt, num_cache
    ); 
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    false_color_mapping<<<dim3(w >> 5, h >> 3), dim3(32, 8)>>>(image, output_buffer, color_map_id, accum_cnt, *min_max);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}

CPT_CPU void DepthTracer::param_setter(const std::vector<char>& bytes) {
    color_map_id = Serializer::get<int>(bytes, 0);
}

CPT_CPU std::vector<uint8_t> DepthTracer::get_image_buffer(bool) const {
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    std::vector<uint8_t> byte_buffer(w * h * 4);
    std::vector<float4> host_floats(w * h);
    size_t copy_size = w * h * sizeof(float4);
    CUDA_CHECK_RETURN(cudaMemcpy(host_floats.data(), output_buffer, copy_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < h; i ++) {
        int base = i * w;
        for (int j = 0; j < w; j ++) {
            int pixel_index = base + j;
            const float4& color = host_floats[pixel_index];
            pixel_index <<= 2;
            byte_buffer[pixel_index + 3] = 255;
            byte_buffer[pixel_index + 0] = to_int_linear(color.x);
            byte_buffer[pixel_index + 1] = to_int_linear(color.y);
            byte_buffer[pixel_index + 2] = to_int_linear(color.z);
        }
    }
    return byte_buffer;
}