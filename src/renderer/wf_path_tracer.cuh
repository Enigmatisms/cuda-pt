/**
 * (W)ave(f)ront (S)imple path tracing with stream multiprocessing
 * This is the updated version, here we opt for a
 * gigantic payload (ray) pool
 * @author Qianyue He
 * @date   2024.6.20 -> 2025.1.18
*/
#pragma once
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <cuda/pipeline>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "renderer/wavefront_pt.cuh"

#include "core/progress.h"
#include "core/emitter.cuh"
#include "core/object.cuh"
#include "core/scene.cuh"
#include "core/stats.h"
#include "renderer/path_tracer.cuh"

class WavefrontPathTracer: public PathTracer {
private:
    // double buffering
    PayLoadBufferSoA payload_buffers[2];
    thrust::device_vector<uint32_t> index_buffers[2];
    uint32_t* idx_buffer[2];    

    const dim3 GRID, BLOCK;
    const int NUM_THREADS;

    // double buffering related
    int _cur_traced_pool;           // index for the current path-traced ray pool
    bool _buffer_ready;             // in case the path tracer gets the lock first

    std::condition_variable _cv;
    std::mutex              _mtx;
    std::thread             _th;
    cudaStream_t            _nb_stream;     // non-blocking stream
    std::atomic<bool>       _rdr_valid;     // stop flag
    std::atomic<uint64_t>   _cam_st;        // camera time stamp: used for discarding stale frames
public:
    WavefrontPathTracer(const Scene& scene);

    ~WavefrontPathTracer() {
        _rdr_valid.store(false);
        {   
            std::lock_guard<std::mutex> ul(_mtx);
            _buffer_ready = false;
        }
        _cv.notify_all();
        if (_th.joinable())
            _th.join();
        else
            std::cerr << "Unknown cause: thread not joinable, this can be erroneous.\n";
        payload_buffers[0].destroy();
        payload_buffers[1].destroy();
        CUDA_CHECK_RETURN(cudaStreamDestroy(_nb_stream));
        printf("[Renderer] Wavefront Path Tracer Object destroyed.\n");
    }
    
    virtual CPT_CPU std::vector<uint8_t> render(
        const MaxDepthParams& md,
        int num_iter = 64,
        bool gamma_correction = true
    ) override;

    virtual CPT_CPU void render_online(
        const MaxDepthParams& md,
        bool gamma_corr = false
    ) override;

    virtual CPT_CPU const float* render_raw(
        const MaxDepthParams& md,
        bool gamma_corr = false
    ) override;

    CPT_CPU void prepare_for_rendering() override {
        _th = std::thread([this]() {
            double_buffering_thread();
        });
    }

    // initialize a non-detached thread and call the 
    // raygen_primary_hit shader on a different stream
    CPT_CPU void double_buffering_thread();

    CPT_CPU void update_camera(const DeviceCamera* const cam) override {
        // I have to use .time_since_epoch().count(), though it is not a good practice
        _cam_st.store(std::chrono::steady_clock::now().time_since_epoch().count());
        {
            std::lock_guard<std::mutex> local(_mtx);
            _buffer_ready = false;
        }
        CUDA_CHECK_RETURN(cudaMemcpyAsync(camera, cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice, _nb_stream));
    }
};
