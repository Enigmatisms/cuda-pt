#pragma once
/**
 * @file volume_pt.cuh
 * @author Qianyue He
 * @brief Volume Path Tracer class definition
 * @version 0.1
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 */

#include "core/medium.cuh"
#include "renderer/path_tracer.cuh"

class VolumePathTracer: public PathTracer {
protected:
    int cam_vol_id;
    const Medium** media;
public:
    VolumePathTracer(const Scene& scene);

    ~VolumePathTracer();

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
};