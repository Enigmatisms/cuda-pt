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
#include "renderer/depth.cuh"
#include <cuda/pipeline>

class BVHCostVisualizer : public DepthTracer {
  private:
    int *reduced_max;
    int cost_map_id;
    float max_v;

  public:
    CPT_CPU BVHCostVisualizer(const Scene &scene);

    virtual ~BVHCostVisualizer();

    virtual CPT_CPU void render_online(const MaxDepthParams &md,
                                       bool gamma_corr = false) override;

    virtual CPT_CPU const float *render_raw(const MaxDepthParams &md,
                                            bool gamma_corr = false) override;

    CPT_CPU void param_setter(const std::vector<char> &bytes) override;

    // BVH cost visualizer does not support variance buffer (variance can not be
    // estimated online)
    CPT_CPU const float *get_variance_buffer() const override {
        return nullptr;
    }
};
