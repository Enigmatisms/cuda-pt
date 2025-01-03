/**
 * @file path_tracer.cu
 * @author Qianyue He
 * @brief Megakernel Path Tracer implementation
 * @date 2024.10.10
 * @copyright Copyright (c) 2024
 */

#include "renderer/light_tracer.cuh"

static constexpr int SEED_SCALER = 11451;       //-4!
static constexpr int SHFL_THREAD_X = 4;     // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16 threads
static constexpr int SHFL_THREAD_Y = 3;     // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16 threads

CPT_CPU std::vector<uint8_t> LightTracer::render(
    const MaxDepthParams& md,
    int num_iter,
    bool gamma_correction
) {
    printf("Rendering starts.\n");
    TicToc _timer("render_lt_kernel()", num_iter);
    size_t cached_size = 2 * num_cache * sizeof(float4);
    for (int i = 0; i < num_iter; i++) {
        // for more sophisticated renderer (like path tracer), shared_memory should be used
        if (bidirectional) {
            render_pt_kernel<true><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
                *camera, verts, norms, uvs, obj_info, emitter_prims,
                bvh_leaves, nodes, _cached_nodes, image, md,
                output_buffer, num_prims, num_objs, num_emitter, 
                accum_cnt * SEED_SCALER, num_nodes, accum_cnt
            ); 
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }
        render_lt_kernel<false><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
            *camera, verts, norms, uvs, obj_info,
            emitter_prims, bvh_leaves, nodes, _cached_nodes, 
            image, md, nullptr, num_prims, num_objs, num_emitter, 
            i * SEED_SCALER, num_nodes, spec_constraint
        ); 
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printProgress(i, num_iter);
    }
    printf("\n");
    return image.export_cpu(1.f / num_iter, gamma_correction, true);
}

CPT_CPU void LightTracer::render_online(
    const MaxDepthParams& md,
    bool gamma_corr
) {
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0, cached_size = 2 * num_cache * sizeof(float4);
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));

    accum_cnt ++;
    if (bidirectional) {
        render_pt_kernel<false><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
            *camera, verts, norms, uvs, obj_info, emitter_prims,
            bvh_leaves, nodes, _cached_nodes, image, md,
            output_buffer, num_prims, num_objs, num_emitter, 
            accum_cnt * SEED_SCALER, num_nodes, accum_cnt, num_cache, false
        ); 
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    render_lt_kernel<true><<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
        *camera, verts, norms, uvs, obj_info, emitter_prims,
        bvh_leaves, nodes, _cached_nodes, image, md, 
        output_buffer, num_prims, num_objs, num_emitter, 
        accum_cnt * SEED_SCALER, num_nodes, accum_cnt, 
        num_cache, spec_constraint, caustic_scaling, gamma_corr
    ); 
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}