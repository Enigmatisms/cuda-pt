// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * @author: Qianyue He
 * @brief Simple tile-based path tracer
 * @date: 2024.5.12
 */
#pragma once
#include "core/progress.h"
#include "core/scene.cuh"
#include "core/stats.h"
#include "renderer/megakernel_pt.cuh"
#include "renderer/tracer_base.cuh"
#include <cuda/pipeline>
#include <cuda_gl_interop.h>

template <typename Scheduler> class PathTracer : public TracerBase {
  private:
    const bool verbose;

  protected:
    int *_obj_idxs;
    float4 *_nodes;

    CompactedObjInfo *obj_info;
    const int num_objs;
    const int num_nodes;
    const int num_cache; // number of cached BVH nodes
    const int num_emitter;
    const int envmap_id;

    cudaTextureObject_t bvh_leaves;
    cudaTextureObject_t nodes;
    uint4 *_cached_nodes;

    int *emitter_prims;

  public:
    PathTracer(const Scene &scene, bool _verbose = true);

    virtual ~PathTracer();

    virtual CPT_CPU std::vector<uint8_t>
    render(const MaxDepthParams &md, int num_iter = 64,
           bool gamma_correction = true) override;

    virtual CPT_CPU void render_online(const MaxDepthParams &md,
                                       bool gamma_corr = false) override;

    virtual CPT_CPU const float *render_raw(const MaxDepthParams &md,
                                            bool gamma_corr = false) override;
};
