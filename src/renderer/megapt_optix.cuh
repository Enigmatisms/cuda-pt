/**
 * @file megapt_optix.cuh
 * @author Qianyue He
 * @brief OptiX hardware accelerated renderer
 * @version 0.1
 * @date 2024-12-18
 * @copyright Copyright (c) 2024
 */
#pragma once

#include <cuda/pipeline>
#include <cuda_gl_interop.h>
#include "core/stats.h"
#include "core/scene.cuh"
#include "core/progress.h"
#include "renderer/tracer_base.cuh"
#include "renderer/optix_pt_kernel.cuh"

#include "optix/sbt.cuh"
#include "optix/optix_utils.cuh"

class PathTracerOptiX: public TracerBase {
private:
    int* _obj_idxs;         // primitive to object mapping
protected:
    // General megakernel logic
    ObjInfo* obj_info;
    int num_objs;
    int num_emitter;

    int* emitter_prims;
    int accum_cnt;

    // OptiX related
    cudaTextureObject_t obj_idxs;
    PathTracerStates states;
public:
    PathTracerOptiX(const Scene& scene);

    virtual ~PathTracerOptiX();

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
