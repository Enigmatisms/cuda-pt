/**
 * @file path_tracer.cu
 * @author Qianyue He
 * @brief Megakernel Path Tracer implementation
 * @date 2024.10.10
 * @copyright Copyright (c) 2024
 */

#include "renderer/megapt_optix.cuh"

static constexpr int SEED_SCALER = 11451;       //-4!
static constexpr int SHFL_THREAD_X = 4;     // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16 threads
static constexpr int SHFL_THREAD_Y = 3;     // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16 threads

PathTracerOptiX::PathTracerOptiX(
    const Scene& scene
): TracerBase(scene, true), num_objs(scene.objects.size()), num_emitter(scene.num_emitters)
{
    OPTIX_CHECK(optixInit());
    size_t emitter_prim_size = sizeof(int) * scene.emitter_prims.size();

    // TODO: note that current implementation does not support sphere
    CUDA_CHECK_RETURN(cudaMalloc(&_obj_idxs,  scene.num_prims * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMallocManaged(&obj_info, num_objs * sizeof(ObjInfo)));
    CUDA_CHECK_RETURN(cudaMalloc(&camera, sizeof(DeviceCamera)));
    CUDA_CHECK_RETURN(cudaMalloc(&emitter_prims, emitter_prim_size));
    CUDA_CHECK_RETURN(cudaMemcpy(camera, &scene.cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(emitter_prims, scene.emitter_prims.data(), emitter_prim_size, cudaMemcpyHostToDevice));

    createTexture1D<int>(scene.obj_idxs.data(), num_prims, _obj_idxs, obj_idxs);

    for (int i = 0; i < num_objs; i++)
        obj_info[i] = scene.objects[i];
    printf("[ATTENTION] Note that OptiX Path Tracer is currently triangle-only. Please make sure there is no sphere primitive in the scene.\n");
    // after initializing OptiX, we should create and load the shaders
    printf("[OptiX] Creating programe group...\n");
    create_program_group(states, scene.optix_ptx_path, false);
    printf("[OptiX] Creating OptiX rendering pipeline...\n");
    create_pipeline(states);
    printf("[OptiX] Building Acceleration structure...\n");
    build_accel(states, reinterpret_cast<float4*>(verts.data), verts.size);
    printf("[OptiX] Building Acceleration structure...\n");
    create_sbt(states);
    printf("[OptiX] All preparation steps are done.\n");

    LaunchParams host_params{states.gas_handle};
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(params, &host_params, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

PathTracerOptiX::~PathTracerOptiX() {
    CUDA_CHECK_RETURN(cudaFree(obj_info));
    CUDA_CHECK_RETURN(cudaFree(camera));
    CUDA_CHECK_RETURN(cudaFree(emitter_prims));
    CUDA_CHECK_RETURN(cudaDestroyTextureObject(obj_idxs));
    CUDA_CHECK_RETURN(cudaFree(_obj_idxs));
    printf("[Renderer] OptiX Path Tracer Object destroyed.\n");
}

CPT_CPU std::vector<uint8_t> PathTracerOptiX::render(
    int num_iter,
    int max_depth,
    bool gamma_correction
) {
    printf("Rendering starts.\n");
    TicToc _timer("render_pt_kernel()", num_iter);
    for (int i = 0; i < num_iter; i++) {
        accum_cnt ++;
        render_optix_kernel<false><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y)>>>(
            *camera, verts, norms, uvs, obj_info, obj_idxs,
            emitter_prims, image, output_buffer, num_emitter, 
            i * SEED_SCALER, max_depth, accum_cnt, false
        ); 
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printProgress(i, num_iter);
    }
    printf("\n");
    return image.export_cpu(1.f / num_iter, gamma_correction);
}

CPT_CPU void PathTracerOptiX::render_online(
    int max_depth,
    bool gamma_corr
) {
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    // if we have an illegal memory access here: check whether you have a valid emitter in the xml scene description file.
    // it might be possible that having no valid emitter triggers an illegal memory access
    size_t _num_bytes = 0;
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));

    accum_cnt ++;
    render_optix_kernel<true><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y)>>>(
        *camera, verts, norms, uvs, obj_info, obj_idxs,
        emitter_prims, image, output_buffer, num_emitter, 
        accum_cnt * SEED_SCALER, max_depth, accum_cnt, gamma_corr
    ); 

    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}