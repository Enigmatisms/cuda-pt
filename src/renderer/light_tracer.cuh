/**
 * Light tracing for caustics rendering
 * @date: 9.28.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include <cuda_gl_interop.h>
#include "core/stats.h"
#include "core/scene.cuh"
#include "core/progress.h"
#include "renderer/path_tracer.cuh"

class LightTracer: public PathTracer {
private:
    bool bidirectional;         // whether to use both PT and LT in a single renderer
    int spec_constraint;
    float caustic_scaling;
public:
    /**
     * @param spec_constraint number of specular bounces required for a path to be recorded
     * @param bidir           If true, the backward path tracing will be interleaved
     * @param caustics_scale  scale of the light tracing (to make the light tracing result brighter or darker)
    */
    LightTracer(
        const Scene& scene,
        int spec_constraint,
        bool bidir = false,
        float caustics_scale = 1.f
    ): PathTracer(scene), 
        spec_constraint(spec_constraint), 
        bidirectional(bidir),
        caustic_scaling(caustics_scale) {}

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

    // light tracer does not support variance buffer (variance can not be estimated online)
    CPT_CPU const float* get_variance_buffer() const override {
        return nullptr;
    }
};
