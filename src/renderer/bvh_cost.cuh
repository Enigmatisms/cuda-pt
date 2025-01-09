/**
 * Simple tile-based depth renderer
 * @date: 5.6.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include "core/stats.h"
#include "renderer/depth.cuh"

class BVHCostVisualizer: public DepthTracer {
public:
    BVHCostVisualizer(const Scene& scene);

    virtual ~BVHCostVisualizer();

    virtual CPT_CPU void render_online(
        const MaxDepthParams& md,
        bool gamma_corr = false
    ) override;
};


