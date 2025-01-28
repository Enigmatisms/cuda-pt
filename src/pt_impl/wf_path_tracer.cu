/**
 * @file wf_path_tracer.cu
 * @author Qianyue He
 * @brief Wavefront Path Tracer implementation
 * @date 2024.10.10
 * @copyright Copyright (c) 2024
 */

#include <thrust/binary_search.h>
#include "renderer/wf_path_tracer.cuh"

static constexpr int SHFL_THREAD_X = 5;     // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16 threads
static constexpr int SHFL_THREAD_Y = 2;     // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16 threads
static constexpr int BLOCK_LIMIT = 12;      // Note: Related to occupancy, should be obtained from profiler

WavefrontPathTracer::WavefrontPathTracer(const Scene& scene): 
    PathTracer(scene),
    GRID(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), 
    BLOCK(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y),
    NUM_THREADS(1 << (SHFL_THREAD_X + SHFL_THREAD_Y))
{
    int image_size = image.w() * image.h();
    payload_buffer.init(image_size);
    index_buffer.resize(image_size);
    idx_buffer = thrust::raw_pointer_cast(index_buffer.data());

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    WAVE_SIZE = prop.multiProcessorCount * BLOCK_LIMIT;
}

CPT_CPU void WavefrontPathTracer::render_online(
    const MaxDepthParams& md,
    bool gamma_corr
) {
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0, cached_size = num_cache * sizeof(uint4);
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));

    accum_cnt ++;
    // step1: ray generator, generate the rays and store them in the PayLoadBuffer of a stream
    raygen_primary_hit_shader<<<GRID, BLOCK, cached_size>>>(
        *camera, payload_buffer, verts, norms, uvs,
        obj_info, bvh_leaves, nodes, _cached_nodes, 
        idx_buffer, image.w(), num_nodes, 
        num_cache, accum_cnt, seed_offset, envmap_id
    );

    int num_valid_ray = w * h, max_grid_num = (num_valid_ray + NUM_THREADS - 1) / NUM_THREADS;
    for (int bounce = 0; bounce < md.max_depth; bounce ++) {
#ifdef NO_RAY_SORTING
        num_valid_ray = partition_func(
            index_buffer.begin(), 
            index_buffer.begin() + num_valid_ray,
            ActiveRayFunctor()
        ) - index_buffer.begin();
#else
        thrust::sort(
            index_buffer.begin(), index_buffer.begin() + num_valid_ray
        );
        num_valid_ray = thrust::lower_bound(index_buffer.begin(), 
                        index_buffer.begin() + num_valid_ray, 0x80000000) - index_buffer.begin();
#endif  // NO_RAY_SORTING

        // here, if after partition, there is no valid PathPayLoad in the buffer, then we break from the for loop
        if (!num_valid_ray) {
            break;
        }
        int NUM_GRID = (num_valid_ray + NUM_THREADS - 1) / NUM_THREADS;
        NUM_GRID = padded_grid(NUM_GRID, max_grid_num);

        fused_ray_bounce_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
            payload_buffer, verts, norms, uvs, obj_info, emitter_prims,
            bvh_leaves, nodes, _cached_nodes, idx_buffer, 
            num_emitter, num_nodes, num_cache, bounce > 0
        );

        if (bounce + 1 >= md.max_depth) break;
        fused_closesthit_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
            payload_buffer, verts, norms, uvs, 
            bvh_leaves, nodes, _cached_nodes, idx_buffer, 
            num_nodes, num_cache, bounce, envmap_id
        );
    }

    radiance_splat<true><<<GRID, BLOCK>>>(
        payload_buffer, image, output_buffer, var_buffer, accum_cnt, gamma_corr
    );
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}

CPT_CPU std::vector<uint8_t> WavefrontPathTracer::render(
    const MaxDepthParams& md,
    int num_iter,
    bool gamma_correction
) {
    TicToc _timer("render_pt_kernel()", num_iter);
    
    // step 1, allocate 2D array of CUDA memory to hold: PathPayLoad
    uint32_t* const ray_idx_buffer = thrust::raw_pointer_cast(index_buffer.data()),
                cached_size = num_cache * sizeof(uint4);

    int num_valid_ray = w * h, max_grid_num = (num_valid_ray + NUM_THREADS - 1) / NUM_THREADS;
    for (int i = 0; i < num_iter; i++) {
        // step1: ray generator, generate the rays and store them in the PayLoadBuffer of a stream
        raygen_primary_hit_shader<<<GRID, BLOCK, cached_size>>>(
            *camera, payload_buffer, verts, norms, uvs,
            obj_info, bvh_leaves, nodes, _cached_nodes, 
            idx_buffer, image.w(), num_nodes, 
            num_cache, i, seed_offset, envmap_id
        );
        for (int bounce = 0; bounce < md.max_depth; bounce ++) {
#ifdef NO_RAY_SORTING
            num_valid_ray = partition_func(
                index_buffer.begin(), 
                index_buffer.begin() + num_valid_ray,
                ActiveRayFunctor()
            ) - index_buffer.begin();
#else
            thrust::sort(
                index_buffer.begin(), index_buffer.begin() + num_valid_ray
            );
            num_valid_ray = thrust::lower_bound(index_buffer.begin(), 
                            index_buffer.begin() + num_valid_ray, 0x80000000) - index_buffer.begin();
#endif  // NO_RAY_SORTING

            // here, if after partition, there is no valid PathPayLoad in the buffer, then we break from the for loop
            if (!num_valid_ray) break;
            int NUM_GRID = (num_valid_ray + NUM_THREADS - 1) / NUM_THREADS;
            NUM_GRID = padded_grid(NUM_GRID, max_grid_num);

            fused_ray_bounce_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
                payload_buffer, verts, norms, uvs, obj_info, emitter_prims,
                bvh_leaves, nodes, _cached_nodes, idx_buffer, 
                num_emitter, num_nodes, num_cache, bounce > 0
            );

            if (bounce + 1 >= md.max_depth) break;
            fused_closesthit_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
                payload_buffer, verts, norms, uvs, 
                bvh_leaves, nodes, _cached_nodes, idx_buffer, 
                num_nodes, num_cache, bounce, envmap_id
            );
        }

        radiance_splat<false><<<GRID, BLOCK>>>(
            payload_buffer, image, output_buffer, var_buffer, accum_cnt, gamma_correction
        );
        // should we synchronize here? Yes, host end needs this
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printProgress(i, num_iter);
        num_valid_ray = w * h;
    }
    printf("\n");
    // TODO: wavefront path tracing does not support online visualization yet
    return image.export_cpu(1.f / num_iter, gamma_correction);
}

CPT_CPU const float* WavefrontPathTracer::render_raw(
    const MaxDepthParams& md,
    bool gamma_corr
) {
    size_t cached_size = num_cache * sizeof(uint4);
    accum_cnt ++;

    // step1: ray generator, generate the rays and store them in the PayLoadBuffer of a stream
    raygen_primary_hit_shader<<<GRID, BLOCK, cached_size>>>(
        *camera, payload_buffer, verts, norms, uvs,
        obj_info, bvh_leaves, nodes, _cached_nodes, 
        idx_buffer, image.w(), num_nodes, 
        num_cache, accum_cnt, seed_offset, envmap_id
    );

    int num_valid_ray = w * h, max_grid_num = (num_valid_ray + NUM_THREADS - 1) / NUM_THREADS;
    for (int bounce = 0; bounce < md.max_depth; bounce ++) {
        // step4: thrust stream compaction (optional)
#ifdef NO_RAY_SORTING
        num_valid_ray = partition_func(
            index_buffer.begin(), 
            index_buffer.begin() + num_valid_ray,
            ActiveRayFunctor()
        ) - index_buffer.begin();
#else
        thrust::sort(
            index_buffer.begin(), index_buffer.begin() + num_valid_ray
        );
        num_valid_ray = thrust::lower_bound(index_buffer.begin(), 
                        index_buffer.begin() + num_valid_ray, 0x80000000) - index_buffer.begin();
#endif  // NO_RAY_SORTING

        // here, if after partition, there is no valid PathPayLoad in the buffer, then we break from the for loop
        if (!num_valid_ray) break;
        int NUM_GRID = (num_valid_ray + NUM_THREADS - 1) / NUM_THREADS;
        NUM_GRID = padded_grid(NUM_GRID, max_grid_num);

        // step5: NEE shader
        fused_ray_bounce_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
            payload_buffer, verts, norms, uvs, obj_info, emitter_prims,
            bvh_leaves, nodes, _cached_nodes, idx_buffer, 
            num_emitter, num_nodes, num_cache, bounce > 0
        );

        if (bounce + 1 >= md.max_depth) break;
        fused_closesthit_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
            payload_buffer, verts, norms, uvs, 
            bvh_leaves, nodes, _cached_nodes, idx_buffer, 
            num_nodes, num_cache, bounce, envmap_id
        );
    }

    // step8: accumulating radiance to the rgb buffer
    radiance_splat<true><<<GRID, BLOCK>>>(
        payload_buffer, image, output_buffer, var_buffer, accum_cnt, gamma_corr
    );
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return output_buffer;
}