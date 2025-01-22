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

WavefrontPathTracer::WavefrontPathTracer(const Scene& scene): 
    PathTracer(scene),
    GRID(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y), 
    BLOCK(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y),
    NUM_THREADS(1 << (SHFL_THREAD_X + SHFL_THREAD_Y)),
    _cur_traced_pool(0),
    _buffer_ready(false)
{
    int image_size = image.w() * image.h();
    payload_buffers[0].init(image_size);
    payload_buffers[1].init(image_size);
    index_buffers[0].resize(image_size);
    index_buffers[1].resize(image_size);
    idx_buffer[0] = thrust::raw_pointer_cast(index_buffers[0].data());
    idx_buffer[1] = thrust::raw_pointer_cast(index_buffers[1].data());
    CUDA_CHECK_RETURN(cudaStreamCreateWithFlags(&_nb_stream, cudaStreamNonBlocking));
    _rdr_valid.store(true);
}

CPT_CPU void WavefrontPathTracer::double_buffering_thread() {
    const size_t cached_size = num_cache * sizeof(uint4);
    int cur_traced_pool = 0;
    for(int local_cnt; _rdr_valid.load(); local_cnt ++) {
        uint64_t local_st = _cam_st.load();
        raygen_primary_hit_shader<<<GRID, BLOCK, cached_size, _nb_stream>>>(
            *camera, payload_buffers[cur_traced_pool], verts, norms, uvs,
            obj_info, bvh_leaves, nodes, _cached_nodes, 
            idx_buffer[cur_traced_pool], num_prims, image.w(), 
            num_nodes, num_cache, local_cnt, seed_offset
        );
        CUDA_CHECK_RETURN(cudaStreamSynchronize(_nb_stream));
        /**
         * If we don't check the timestamp, it will be very likely that
         * another camera update event occurs during `raygen_primary_hit_shader`
         * and of course, the current frame will be stale. If we don't discard it
         * there will be some phantom on the rendered image. If there is no camera update
         * during the `raygen_primary_hit_shader`, the following load will check out
         */
        if (_cam_st.load() != local_st) continue;       // bypass the stale frame
        std::lock_guard<std::mutex> lock(_mtx);
        _cur_traced_pool = cur_traced_pool;
        _buffer_ready = true;
        _cv.notify_one();
        cur_traced_pool = 1 - cur_traced_pool;
    }
}

CPT_CPU void WavefrontPathTracer::render_online(
    const MaxDepthParams& md,
    bool gamma_corr
) {
    std::unique_lock<std::mutex> ul(_mtx);
    _cv.wait(ul, [this]{ return _buffer_ready || !_rdr_valid.load(); });
    if (!_rdr_valid.load()) return;

    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0, cached_size = num_cache * sizeof(uint4);
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));

    int num_valid_ray = w * h, cur_traced_pool = _cur_traced_pool;
    auto index_buffer = index_buffers[cur_traced_pool];
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

        fused_ray_bounce_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
            payload_buffers[cur_traced_pool], verts, norms, uvs, obj_info, 
            emitter_prims, bvh_leaves, nodes, _cached_nodes, idx_buffer[cur_traced_pool], 
            num_prims, num_objs, num_emitter, num_nodes, num_cache, bounce > 0
        );

        if (bounce + 1 >= md.max_depth) break;
        fused_closesthit_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
            payload_buffers[cur_traced_pool], verts, norms, uvs, 
            bvh_leaves, nodes, _cached_nodes, idx_buffer[cur_traced_pool], 
            num_prims, num_nodes, num_cache, bounce, envmap_id
        );
    }

    accum_cnt ++;
    radiance_splat<true><<<GRID, BLOCK>>>(
        payload_buffers[_cur_traced_pool], image, 
        output_buffer, var_buffer, accum_cnt, gamma_corr
    );
    ul.unlock();
    _buffer_ready = false;
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}

CPT_CPU std::vector<uint8_t> WavefrontPathTracer::render(
    const MaxDepthParams& md,
    int num_iter,
    bool gamma_correction
) {
    TicToc _timer("render_pt_kernel()", num_iter);
    
    uint32_t cached_size = num_cache * sizeof(uint4);
    for (int i = 0; i < num_iter; i++) {
        int num_valid_ray = w * h;

        std::unique_lock<std::mutex> ul(_mtx);
        _cv.wait(ul, [this]{ return _buffer_ready || !_rdr_valid.load(); });
        if (!_rdr_valid.load()) break;

        for (int bounce = 0; bounce < md.max_depth; bounce ++) {
            auto index_buffer = index_buffers[_cur_traced_pool];
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

            if (!num_valid_ray) break;
            int NUM_GRID = (num_valid_ray + NUM_THREADS - 1) / NUM_THREADS;

            fused_ray_bounce_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
                payload_buffers[_cur_traced_pool], verts, norms, uvs, obj_info, emitter_prims,
                bvh_leaves, nodes, _cached_nodes, idx_buffer[_cur_traced_pool], 
                num_prims, num_objs, num_emitter, num_nodes, num_cache, bounce > 0
            );

            if (bounce + 1 >= md.max_depth) break;
            fused_closesthit_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
                payload_buffers[_cur_traced_pool], verts, norms, uvs, 
                bvh_leaves, nodes, _cached_nodes, idx_buffer[_cur_traced_pool], 
                num_prims, num_nodes, num_cache, bounce, envmap_id
            );
        }

        radiance_splat<false><<<GRID, BLOCK>>>(
            payload_buffers[_cur_traced_pool], image, output_buffer, var_buffer, i, gamma_correction
        );
        CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStreamDefault));

        _buffer_ready = false;
        ul.unlock();
        printProgress(i, num_iter);
    }
    printf("\n");
    // TODO: wavefront path tracing does not support online visualization yet
    return image.export_cpu(1.f / num_iter, gamma_correction);
}

CPT_CPU const float* WavefrontPathTracer::render_raw(
    const MaxDepthParams& md,
    bool gamma_corr
) {
    const size_t cached_size = num_cache * sizeof(uint4);

    std::unique_lock<std::mutex> ul(_mtx);
    _cv.wait(ul, [this]{ return _buffer_ready || !_rdr_valid.load(); });
    if (!_rdr_valid.load()) return output_buffer;

    accum_cnt ++;
    int num_valid_ray = w * h;
    for (int bounce = 0; bounce < md.max_depth; bounce ++) {
        auto index_buffer = index_buffers[_cur_traced_pool];
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

        // step5: NEE shader
        fused_ray_bounce_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
            payload_buffers[_cur_traced_pool], verts, norms, uvs, obj_info, emitter_prims,
            bvh_leaves, nodes, _cached_nodes, idx_buffer[_cur_traced_pool], 
            num_prims, num_objs, num_emitter, num_nodes, num_cache, bounce > 0
        );

        if (bounce + 1 >= md.max_depth) break;
        fused_closesthit_shader<<<NUM_GRID, NUM_THREADS, cached_size>>>(
            payload_buffers[_cur_traced_pool], verts, norms, uvs, 
            bvh_leaves, nodes, _cached_nodes, idx_buffer[_cur_traced_pool], 
            num_prims, num_nodes, num_cache, bounce, envmap_id
        );
    }

    // step8: accumulating radiance to the rgb buffer
    radiance_splat<true><<<GRID, BLOCK>>>(
        payload_buffers[_cur_traced_pool], image, output_buffer, var_buffer, accum_cnt, gamma_corr
    );

    CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStreamDefault));
    _buffer_ready = false;
    return output_buffer;
}