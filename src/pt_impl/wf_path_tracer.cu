/**
 * @file wf_path_tracer.cu
 * @author Qianyue He
 * @brief Wavefront Path Tracer implementation
 * @date 2024.10.10
 * @copyright Copyright (c) 2024
 */

#include "renderer/wf_path_tracer.cuh"

WavefrontPathTracer::WavefrontPathTracer(
        const Scene& scene,
        const PrecomputedArray& _verts,
        const ArrayType<Vec3>& _norms, 
        const ArrayType<Vec2>& _uvs,
        int num_emitter
): PathTracer(scene, _verts, _norms, _uvs, num_emitter),
    x_patches(w / PATCH_X), y_patches(h / PATCH_Y),
    num_patches(x_patches * y_patches), 
    GRID(BLOCK_X, BLOCK_Y), 
    BLOCK(THREAD_X, THREAD_Y)
{
    for (int i = 0; i < NUM_STREAM; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamDefault);
    payload_buffer.init(NUM_STREAM * PATCH_X, PATCH_Y);
    index_buffer.resize(NUM_STREAM * TOTAL_RAY);
    ray_idx_buffer = thrust::raw_pointer_cast(index_buffer.data());
}

CPT_CPU void WavefrontPathTracer::render_online(
    int max_depth
) {
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0, cached_size = 2 * num_cache * sizeof(float4);
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));

    accum_cnt ++;

    #pragma omp parallel for num_threads(NUM_STREAM)
    for (int p_idx = 0; p_idx < num_patches; p_idx++) {
        int patch_x = p_idx % x_patches, patch_y = p_idx / x_patches, stream_id = omp_get_thread_num();
        int stream_offset = stream_id * TOTAL_RAY;
        auto cur_stream = streams[stream_id];

        // step1: ray generator, generate the rays and store them in the PayLoadBuffer of a stream
        raygen_primary_hit_shader<<<GRID, BLOCK, cached_size, cur_stream>>>(
            *camera, *verts, payload_buffer, obj_info, aabbs, 
            norms, uvs, bvh_leaves, node_fronts, node_backs, 
            _cached_nodes, ray_idx_buffer, stream_offset, num_prims, 
            patch_x, patch_y, accum_cnt, stream_id, image.w(), num_nodes, num_cache);
        int num_valid_ray = TOTAL_RAY;
        auto start_iter = index_buffer.begin() + stream_id * TOTAL_RAY;
        for (int bounce = 0; bounce < max_depth; bounce ++) {
            
            // TODO: we can implement a RR shader here.
            // step3: miss shader (ray inactive)
#ifndef FUSED_MISS_SHADER
            miss_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                payload_buffer, ray_idx_buffer, stream_offset, num_valid_ray
            );
#endif  // FUSED_MISS_SHADER
            
            // step4: thrust stream compaction (optional)
#ifndef NO_STREAM_COMPACTION
            num_valid_ray = partition_func(
                thrust::cuda::par.on(cur_stream), 
                start_iter, start_iter + num_valid_ray,
                ActiveRayFunctor(payload_buffer.ray_ds, NUM_STREAM * PATCH_X)
            ) - start_iter;
#else
            num_valid_ray = TOTAL_RAY;    
#endif  // NO_STREAM_COMPACTION

#ifndef NO_RAY_SORTING
            // sort the ray (indices) by their ray tag (hit object)
            // ray sorting is extremely slow
            thrust::sort(
                thrust::cuda::par.on(cur_stream), 
                start_iter, start_iter + num_valid_ray,
                RaySortFunctor(payload_buffer.ray_ds, NUM_STREAM * PATCH_X)
            );
#endif

            // here, if after partition, there is no valid PathPayLoad in the buffer, then we break from the for loop
            if (!num_valid_ray) break;

            // step5: NEE shader
            nee_shader<<<GRID, BLOCK, cached_size, cur_stream>>>(
                *verts, payload_buffer, obj_info, aabbs, norms, uvs, bvh_leaves, 
                node_fronts, node_backs, _cached_nodes, ray_idx_buffer, 
                stream_offset, num_prims, num_objs, num_emitter, num_valid_ray, num_nodes, num_cache
            );

            // step6: emission shader + ray update shader
            bsdf_local_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                payload_buffer, obj_info, aabbs, uvs, ray_idx_buffer,
                stream_offset, num_prims, num_valid_ray, bounce > 0
            );

            // step2: closesthit shader
            if (bounce + 1 >= max_depth) break;
            closesthit_shader<<<GRID, BLOCK, cached_size, cur_stream>>>(
                *verts, payload_buffer, obj_info, aabbs, norms, uvs, 
                bvh_leaves, node_fronts, node_backs, _cached_nodes,
                ray_idx_buffer, stream_offset, num_prims, num_valid_ray, num_nodes, num_cache
            );
        }

        // step8: accumulating radiance to the rgb buffer
        radiance_splat<true><<<GRID, BLOCK, 0, cur_stream>>>(
            payload_buffer, image, stream_id, patch_x, patch_y, accum_cnt, output_buffer
        );
    }
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}

CPT_CPU std::vector<uint8_t> WavefrontPathTracer::render(
    int num_iter,
    int max_depth,
    bool gamma_correction
) {
    TicToc _timer("render_pt_kernel()", num_iter);
    
    const int x_patches = w / PATCH_X, y_patches = h / PATCH_Y;
    const int num_patches = x_patches * y_patches;
    // step 1, allocate 2D array of CUDA memory to hold: PathPayLoad
    uint32_t* const ray_idx_buffer = thrust::raw_pointer_cast(index_buffer.data()),
                cached_size = 2 * num_cache * sizeof(float4);
    const dim3 GRID(BLOCK_X, BLOCK_Y), BLOCK(THREAD_X, THREAD_Y);

    for (int i = 0; i < num_iter; i++) {
        // here, we should use multi threading to submit the kernel call
        // each thread is responsible for only one stream (and dedicated to that stream only)
        // If we decide to use 8 streams, then we will use 8 CPU threads
        // Using multi-threading to submit kernel, we can avoid stucking on just one stream
        // This can be extended even further: use a high performance thread pool
        #pragma omp parallel for num_threads(NUM_STREAM)
        for (int p_idx = 0; p_idx < num_patches; p_idx++) {
            int patch_x = p_idx % x_patches, patch_y = p_idx / x_patches, stream_id = omp_get_thread_num();
            int stream_offset = stream_id * TOTAL_RAY;
            auto cur_stream = streams[stream_id];

            // step1: ray generator, generate the rays and store them in the PayLoadBuffer of a stream
            raygen_primary_hit_shader<<<GRID, BLOCK, cached_size, cur_stream>>>(
                *camera, *verts, payload_buffer, obj_info, aabbs, 
                norms, uvs, bvh_leaves, node_fronts, node_backs, 
                _cached_nodes, ray_idx_buffer, stream_offset, num_prims, 
                patch_x, patch_y, i, stream_id, image.w(), num_nodes, num_cache);
            int num_valid_ray = TOTAL_RAY;
            auto start_iter = index_buffer.begin() + stream_id * TOTAL_RAY;
            for (int bounce = 0; bounce < max_depth; bounce ++) {
                
                // TODO: we can implement a RR shader here.
                // step3: miss shader (ray inactive)
#ifndef FUSED_MISS_SHADER
                miss_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                    payload_buffer, ray_idx_buffer, stream_offset, num_valid_ray
                );
#endif  // FUSED_MISS_SHADER
                
                // step4: thrust stream compaction (optional)
#ifndef NO_STREAM_COMPACTION
                num_valid_ray = partition_func(
                    thrust::cuda::par.on(cur_stream), 
                    start_iter, start_iter + num_valid_ray,
                    ActiveRayFunctor(payload_buffer.ray_ds, NUM_STREAM * PATCH_X)
                ) - start_iter;
#else
                num_valid_ray = TOTAL_RAY;    
#endif  // NO_STREAM_COMPACTION

#ifndef NO_RAY_SORTING
                // sort the ray (indices) by their ray tag (hit object)
                // ray sorting is extremely slow
                thrust::sort(
                    thrust::cuda::par.on(cur_stream), 
                    start_iter, start_iter + num_valid_ray,
                    RaySortFunctor(payload_buffer.ray_ds, NUM_STREAM * PATCH_X)
                );
#endif

                // here, if after partition, there is no valid PathPayLoad in the buffer, then we break from the for loop
                if (!num_valid_ray) break;

                // step5: NEE shader
                nee_shader<<<GRID, BLOCK, cached_size, cur_stream>>>(
                    *verts, payload_buffer, obj_info, aabbs, norms, uvs, bvh_leaves, 
                    node_fronts, node_backs, _cached_nodes, ray_idx_buffer, 
                    stream_offset, num_prims, num_objs, num_emitter, num_valid_ray, num_nodes, num_cache
                );

                // step6: emission shader + ray update shader
                bsdf_local_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                    payload_buffer, obj_info, aabbs, uvs, ray_idx_buffer,
                    stream_offset, num_prims, num_valid_ray, bounce > 0
                );

                // step2: closesthit shader
                if (bounce + 1 >= max_depth) break;
                closesthit_shader<<<GRID, BLOCK, cached_size, cur_stream>>>(
                    *verts, payload_buffer, obj_info, aabbs, norms, uvs, 
                    bvh_leaves, node_fronts, node_backs, _cached_nodes,
                    ray_idx_buffer, stream_offset, num_prims, num_valid_ray, num_nodes, num_cache
                );
            }

            // step8: accumulating radiance to the rgb buffer
            radiance_splat<false><<<GRID, BLOCK, 0, cur_stream>>>(
                payload_buffer, image, stream_id, patch_x, patch_y
            );
        }

        // should we synchronize here? Yes, host end needs this
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printProgress(i, num_iter);
    }
    printf("\n");

    // TODO: wavefront path tracing does not support online visualization yet
    return image.export_cpu(1.f / num_iter, gamma_correction);
}