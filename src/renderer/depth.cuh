/**
 * Simple tile-based depth renderer
 * @date: 5.6.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include "core/stats.h"
#include "renderer/tracer_base.cuh"

class DepthTracer: public TracerBase {
private:
    cudaArray_t _colormap_data[3];
    int* _obj_idxs;
    float4* _nodes;
protected:
    int color_map_id;
    int num_nodes;
    int num_cache;                  // number of cached BVH nodes

    uint4* _cached_nodes;
    cudaTextureObject_t colormaps[3];
    cudaTextureObject_t bvh_leaves;
    cudaTextureObject_t nodes;
    int2* min_max;
public:
    DepthTracer(const Scene& scene);

    virtual ~DepthTracer();

    virtual CPT_CPU std::vector<uint8_t> render(
        const MaxDepthParams& md,
        int num_iter = 64,
        bool gamma_correction = true
    ) override;

    virtual CPT_CPU void render_online(
        const MaxDepthParams& md,
        bool gamma_corr = false
    ) override;

    void create_color_map_texture();

    CPT_CPU void param_setter(const std::vector<char>& bytes);

    CPT_CPU std::vector<uint8_t> get_image_buffer(bool gamma_cor) const override;

    virtual CPT_CPU float* render_raw(
        const MaxDepthParams& md,
        bool gamma_corr = false
    ) override;
};

extern CPT_GPU_CONST cudaTextureObject_t COLOR_MAPS[3];
