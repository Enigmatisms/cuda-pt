/**
 * (W)ave(f)ront (S)imple path tracing with stream multiprocessing
 * This is the updated version, here we opt for a
 * gigantic payload (ray) pool
 * @author Qianyue He
 * @date   2024.6.20 -> 2025.1.18
*/
#pragma once
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
    PayLoadBufferSoA payload_buffer;
    thrust::device_vector<uint32_t> index_buffer;
    uint32_t* idx_buffer;    

    const dim3 GRID, BLOCK;
    const int NUM_THREADS;
public:
    WavefrontPathTracer(const Scene& scene);

    ~WavefrontPathTracer() {
        payload_buffer.destroy();
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

    CPT_CPU virtual const float* render_raw(
        const MaxDepthParams& md,
        bool gamma_corr = false
    ) override;
};
