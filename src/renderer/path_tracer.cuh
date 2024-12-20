/**
 * Simple tile-based path tracer
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include <cuda_gl_interop.h>
#include "core/stats.h"
#include "core/scene.cuh"
#include "core/progress.h"
#include "renderer/tracer_base.cuh"
#include "renderer/megakernel_pt.cuh"

class PathTracer: public TracerBase {
private:
    int* _obj_idxs;
    float4* _nodes;
protected:
    ObjInfo* obj_info;
    int num_objs;
    int num_nodes;
    int num_cache;                  // number of cached BVH nodes
    int num_emitter;

    cudaTextureObject_t bvh_leaves;
    cudaTextureObject_t nodes;
    float4* _cached_nodes;

    float* output_buffer;                // output buffer for images
    int* emitter_prims;
public:
    PathTracer(const Scene& scene);

    virtual ~PathTracer();

    virtual CPT_CPU std::vector<uint8_t> render(
        int num_iter = 64,
        int max_depth = 4,
        bool gamma_correction = true
    ) override;

    virtual CPT_CPU void render_online(
        int max_depth = 4,
        bool gamma_corr = false
    ) override;
};
