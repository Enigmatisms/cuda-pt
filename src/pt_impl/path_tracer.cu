/**
 * @file path_tracer.cu
 * @author Qianyue He
 * @brief Megakernel Path Tracer implementation
 * @date 2024.10.10
 * @copyright Copyright (c) 2024
 */

#include "renderer/path_tracer.cuh"

static constexpr int SHFL_THREAD_X = 4;     // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16 threads
static constexpr int SHFL_THREAD_Y = 4;     // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16 threads

PathTracer::PathTracer(
    const Scene& scene,
    const PrecomputedArray& _verts,
    const ArrayType<Vec3>& _norms, 
    const ArrayType<Vec2>& _uvs,
    int num_emitter
): TracerBase(scene.shapes, _verts, _norms, _uvs, scene.config.width, scene.config.height), 
    num_objs(scene.objects.size()), num_nodes(-1), num_emitter(num_emitter), 
    cuda_texture_id(0), pbo_id(0), output_buffer(nullptr), accum_cnt(0)
{
#ifdef RENDERER_USE_BVH
    if (scene.bvh_available()) {
        size_t num_bvh  = scene.bvh_fronts.size();
        num_nodes = scene.node_fronts.size();
        num_cache = scene.cache_fronts.size();
        CUDA_CHECK_RETURN(cudaMalloc(&_bvh_fronts,  num_bvh * sizeof(float4)));
        CUDA_CHECK_RETURN(cudaMalloc(&_bvh_backs,   num_bvh * sizeof(float4)));
        CUDA_CHECK_RETURN(cudaMalloc(&_node_fronts, num_nodes * sizeof(float4)));
        CUDA_CHECK_RETURN(cudaMalloc(&_node_backs,  num_nodes * sizeof(float4)));
        CUDA_CHECK_RETURN(cudaMalloc(&_cached_nodes, 2 * num_cache * sizeof(float4)));
        CUDA_CHECK_RETURN(cudaMalloc(& _node_offsets, num_nodes * sizeof(int)));
        PathTracer::createTexture1D<float4>(scene.bvh_fronts.data(),  num_bvh,   _bvh_fronts,  bvh_fronts);
        PathTracer::createTexture1D<float4>(scene.bvh_backs.data(),   num_bvh,   _bvh_backs,   bvh_backs);
        PathTracer::createTexture1D<float4>(scene.node_fronts.data(), num_nodes, _node_fronts, node_fronts);
        PathTracer::createTexture1D<float4>(scene.node_backs.data(),  num_nodes, _node_backs,  node_backs);
        PathTracer::createTexture1D<int>(scene.node_offsets.data(), num_nodes, _node_offsets, node_offsets);
        CUDA_CHECK_RETURN(cudaMemcpy(_cached_nodes, scene.cache_fronts.data(), sizeof(float4) * num_cache, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(&_cached_nodes[num_cache], scene.cache_backs.data(), sizeof(float4) * num_cache, cudaMemcpyHostToDevice));
    } else {
        throw std::runtime_error("BVH not available in scene. Abort.");
    }
#endif  // RENDERER_USE_BVH

    CUDA_CHECK_RETURN(cudaMallocManaged(&obj_info, num_objs * sizeof(ObjInfo)));
    CUDA_CHECK_RETURN(cudaMalloc(&camera, sizeof(DeviceCamera)));
    CUDA_CHECK_RETURN(cudaMemcpy(camera, &scene.cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice));

    for (int i = 0; i < num_objs; i++)
        obj_info[i] = scene.objects[i];
}

PathTracer::~PathTracer() {
    CUDA_CHECK_RETURN(cudaFree(obj_info));
    CUDA_CHECK_RETURN(cudaFree(camera));
#ifdef RENDERER_USE_BVH
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(bvh_fronts));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(bvh_backs));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(node_fronts));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(node_backs));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(node_offsets));
    CUDA_CHECK_RETURN(cudaFree(_bvh_fronts));
    CUDA_CHECK_RETURN(cudaFree(_bvh_backs));
    CUDA_CHECK_RETURN(cudaFree(_node_fronts));
    CUDA_CHECK_RETURN(cudaFree(_node_backs));
    CUDA_CHECK_RETURN(cudaFree(_node_offsets));
    CUDA_CHECK_RETURN(cudaFree(_cached_nodes));
#endif  // RENDERER_USE_BVH
}

CPT_CPU std::vector<uint8_t> PathTracer::render(
    int num_iter,
    int max_depth,
    bool gamma_correction
) {
    printf("Rendering starts.\n");
    TicToc _timer("render_pt_kernel()", num_iter);
    size_t cached_size = 2 * num_cache * sizeof(float4);
    for (int i = 0; i < num_iter; i++) {
        // for more sophisticated renderer (like path tracer), shared_memory should be used
        render_pt_kernel<false><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
            *camera, *verts, obj_info, aabbs, norms, uvs, 
            bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets,
            _cached_nodes, image, output_buffer, num_prims, num_objs, num_emitter, 
            i * SEED_SCALER, max_depth, num_nodes, accum_cnt, num_cache
        ); 
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printProgress(i, num_iter);
    }
    printf("\n");
    return image.export_cpu(1.f / num_iter, gamma_correction);
}

CPT_CPU void PathTracer::render_online(
    int max_depth
) {
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0, cached_size = 2 * num_cache * sizeof(float4);
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));

    accum_cnt ++;
    render_pt_kernel<true><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
        *camera, *verts, obj_info, aabbs, norms, uvs, 
        bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets,
        _cached_nodes, image, output_buffer, num_prims, num_objs, num_emitter, 
        accum_cnt * SEED_SCALER, max_depth, num_nodes, accum_cnt, num_cache
    ); 
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}