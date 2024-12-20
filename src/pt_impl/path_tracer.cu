/**
 * @file path_tracer.cu
 * @author Qianyue He
 * @brief Megakernel Path Tracer implementation
 * @date 2024.10.10
 * @copyright Copyright (c) 2024
 */

#include "renderer/path_tracer.cuh"

static constexpr int SEED_SCALER = 11451;       //-4!
static constexpr int SHFL_THREAD_X = 4;     // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16 threads
static constexpr int SHFL_THREAD_Y = 3;     // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16 threads

PathTracer::PathTracer(
    const Scene& scene
): TracerBase(scene), 
    num_objs(scene.objects.size()), num_nodes(-1), num_emitter(scene.num_emitters)
{
#ifdef RENDERER_USE_BVH
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
        createTexture1D<int>(scene.obj_idxs.data(), num_bvh, _obj_idxs, bvh_leaves);
        createTexture1D<float4>(scene.nodes.data(), 2 * num_nodes, _nodes, nodes);
        CUDA_CHECK_RETURN(cudaMemcpy(_cached_nodes, scene.cache_fronts.data(), sizeof(float4) * num_cache, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(&_cached_nodes[num_cache], scene.cache_backs.data(), sizeof(float4) * num_cache, cudaMemcpyHostToDevice));
    } else {
        throw std::runtime_error("BVH not available in scene. Abort.");
    }
#endif  // RENDERER_USE_BVH
    size_t emitter_prim_size = sizeof(int) * scene.emitter_prims.size();
    CUDA_CHECK_RETURN(cudaMallocManaged(&obj_info, num_objs * sizeof(ObjInfo)));
    CUDA_CHECK_RETURN(cudaMalloc(&camera, sizeof(DeviceCamera)));
    CUDA_CHECK_RETURN(cudaMalloc(&emitter_prims, emitter_prim_size));
    CUDA_CHECK_RETURN(cudaMemcpy(camera, &scene.cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(emitter_prims, scene.emitter_prims.data(), emitter_prim_size, cudaMemcpyHostToDevice));
    for (int i = 0; i < num_objs; i++)
        obj_info[i] = scene.objects[i];
#ifdef TRIANGLE_ONLY
    printf("[ATTENTION] Note that TRIANGLE_ONLY macro is defined. Please make sure there is no sphere primitive in the scene.\n");
#endif
}

PathTracer::~PathTracer() {
    CUDA_CHECK_RETURN(cudaFree(obj_info));
    CUDA_CHECK_RETURN(cudaFree(camera));
    CUDA_CHECK_RETURN(cudaFree(emitter_prims));
#ifdef RENDERER_USE_BVH
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(bvh_leaves));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(nodes));
    CUDA_CHECK_RETURN(cudaFree(_obj_idxs));
    CUDA_CHECK_RETURN(cudaFree(_nodes));
    CUDA_CHECK_RETURN(cudaFree(_cached_nodes));
#endif  // RENDERER_USE_BVH
    printf("[Renderer] Path Tracer Object destroyed.\n");
}

CPT_CPU std::vector<uint8_t> PathTracer::render(
    int num_iter,
    int max_depth,
    bool gamma_correction
) {
    printf("Rendering starts.\n");
    TicToc _timer("render_pt_kernel()", num_iter);
    size_t cached_size = std::max(2 * num_cache * sizeof(float4), sizeof(float4));
    for (int i = 0; i < num_iter; i++) {
        // for more sophisticated renderer (like path tracer), shared_memory should be used
        render_pt_kernel<false><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
            *camera, verts, norms, uvs, obj_info, aabbs, 
            emitter_prims, bvh_leaves, nodes, _cached_nodes,
            image, output_buffer, num_prims, num_objs, num_emitter, 
            i * SEED_SCALER, max_depth, num_nodes, accum_cnt, num_cache
        ); 
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printProgress(i, num_iter);
    }
    printf("\n");
    return image.export_cpu(1.f / num_iter, gamma_correction);
}

CPT_CPU void PathTracer::render_online(
    int max_depth,
    bool gamma_corr
) {
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0, cached_size = std::max(2 * num_cache * sizeof(float4), sizeof(float4));
    // if we have an illegal memory access here: check whether you have a valid emitter in the xml scene description file.
    // it might be possible that having no valid emitter triggers an illegal memory access
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));

    accum_cnt ++;
    render_pt_kernel<true><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
        *camera, verts, norms, uvs, obj_info, aabbs, 
        emitter_prims, bvh_leaves, nodes, _cached_nodes,
        image, output_buffer, num_prims, num_objs, num_emitter, 
        accum_cnt * SEED_SCALER, max_depth, num_nodes, accum_cnt, num_cache, gamma_corr
    ); 
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}