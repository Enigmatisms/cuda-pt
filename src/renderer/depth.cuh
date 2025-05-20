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
 * @brief Simple tile-based depth renderer
 * @date: 2024.5.6
 */
#pragma once
#include "core/stats.h"
#include "renderer/tracer_base.cuh"
#include <cuda/pipeline>

class DepthTracer : public TracerBase {
  private:
    cudaArray_t _colormap_data[3];
    int *_obj_idxs;
    float4 *_nodes;

  protected:
    int color_map_id;
    int num_nodes;
    int num_cache; // number of cached BVH nodes

    uint4 *_cached_nodes;
    cudaTextureObject_t colormaps[3];
    cudaTextureObject_t bvh_leaves;
    cudaTextureObject_t nodes;
    int2 *min_max;

  public:
    DepthTracer(const Scene &scene);

    virtual ~DepthTracer();

    virtual CPT_CPU std::vector<uint8_t>
    render(const MaxDepthParams &md, int num_iter = 64,
           bool gamma_correction = true) override;

    virtual CPT_CPU void render_online(const MaxDepthParams &md,
                                       bool gamma_corr = false) override;

    void create_color_map_texture();

    CPT_CPU void param_setter(const std::vector<char> &bytes);

    CPT_CPU std::vector<uint8_t>
    get_image_buffer(bool gamma_cor) const override;

    virtual CPT_CPU const float *render_raw(const MaxDepthParams &md,
                                            bool gamma_corr = false) override;

    // depth tracer does not support variance buffer (variance can not be
    // estimated online)
    CPT_CPU const float *get_variance_buffer() const override {
        return nullptr;
    }
};

extern CPT_GPU_CONST cudaTextureObject_t COLOR_MAPS[3];
