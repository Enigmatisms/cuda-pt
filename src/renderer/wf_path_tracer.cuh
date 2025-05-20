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
 * (W)ave(f)ront (S)imple path tracing with stream multiprocessing
 * This is the updated version, here we opt for a
 * gigantic payload (ray) pool
 * @author Qianyue He
 * @date   2024.6.20 -> 2025.1.18
 */
#pragma once
#include "renderer/wavefront_pt.cuh"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>

#include "core/emitter.cuh"
#include "core/object.cuh"
#include "core/progress.h"
#include "core/scene.cuh"
#include "core/stats.h"
#include "renderer/path_tracer.cuh"

class WavefrontPathTracer : public PathTracer {
  private:
    PayLoadBufferSoA payload_buffer;
    thrust::device_vector<uint32_t> index_buffer;
    uint32_t *idx_buffer;

    const dim3 GRID, BLOCK;
    const int NUM_THREADS;

    int WAVE_SIZE;

  public:
    WavefrontPathTracer(const Scene &scene);

    ~WavefrontPathTracer() {
        payload_buffer.destroy();
        printf("[Renderer] Wavefront Path Tracer Object destroyed.\n");
    }

    virtual CPT_CPU std::vector<uint8_t>
    render(const MaxDepthParams &md, int num_iter = 64,
           bool gamma_correction = true) override;

    virtual CPT_CPU void render_online(const MaxDepthParams &md,
                                       bool gamma_corr = false) override;

    virtual CPT_CPU const float *render_raw(const MaxDepthParams &md,
                                            bool gamma_corr = false) override;

    // eliminating tail effect for small grids
    CPT_CPU_INLINE int padded_grid(int num_grid, int max_grid) const {
        return std::min(max_grid,
                        ((num_grid + WAVE_SIZE - 1) / WAVE_SIZE) * WAVE_SIZE);
    }
};
