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
 * @brief Light tracing for caustics rendering
 * @date: 2024.9.28
 */
#pragma once
#include "core/progress.h"
#include "core/scene.cuh"
#include "core/stats.h"
#include "renderer/path_tracer.cuh"
#include <cuda/pipeline>
#include <cuda_gl_interop.h>

class LightTracer : public PathTracer {
  private:
    bool bidirectional; // whether to use both PT and LT in a single renderer
    int spec_constraint;
    float caustic_scaling;

  public:
    /**
     * @param spec_constraint number of specular bounces required for a path to
     * be recorded
     * @param bidir           If true, the backward path tracing will be
     * interleaved
     * @param caustics_scale  scale of the light tracing (to make the light
     * tracing result brighter or darker)
     */
    LightTracer(const Scene &scene, int spec_constraint, bool bidir = false,
                float caustics_scale = 1.f)
        : PathTracer(scene), spec_constraint(spec_constraint),
          bidirectional(bidir), caustic_scaling(caustics_scale) {}

    virtual CPT_CPU std::vector<uint8_t>
    render(const MaxDepthParams &md, int num_iter = 64,
           bool gamma_correction = true) override;

    virtual CPT_CPU void render_online(const MaxDepthParams &md,
                                       bool gamma_corr = false) override;

    virtual CPT_CPU const float *render_raw(const MaxDepthParams &md,
                                            bool gamma_corr = false) override;

    // light tracer does not support variance buffer (variance can not be
    // estimated online)
    CPT_CPU const float *get_variance_buffer() const override {
        return nullptr;
    }
};
