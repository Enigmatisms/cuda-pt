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
private:
    int* reduced_max;
    float max_v;
public:
    CPT_CPU BVHCostVisualizer(const Scene& scene);

    virtual ~BVHCostVisualizer();

    virtual CPT_CPU void render_online(
        const MaxDepthParams& md,
        bool gamma_corr = false
    ) override;

    CPT_CPU void param_setter(const std::vector<char>& bytes) override;
};


