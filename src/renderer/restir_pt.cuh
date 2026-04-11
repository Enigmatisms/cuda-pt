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
 * @brief ReSTIR DI Path Tracer class definition
 * @date: 2025.4.10
 */
#pragma once
#include "core/medium.cuh"
#include "renderer/path_tracer.cuh"
#include "renderer/restir.cuh"

class ReSTIRPathTracer : public PathTracer<SingleTileScheduler> {
  protected:
    int cam_vol_id;
    Medium **media;
    bool enable_temporal; // temporal reuse flag

    ReSTIRReservoir *curr_reservoirs;
    ReSTIRReservoir *prev_reservoirs;
    ReSTIRGBuffer *g_buffer;

  public:
    ReSTIRPathTracer(const Scene &scene, bool temporal = true);

    ~ReSTIRPathTracer();

    virtual CPT_CPU std::vector<uint8_t>
    render(const MaxDepthParams &md, int num_iter = 64,
           bool gamma_correction = true) override;

    virtual CPT_CPU void render_online(const MaxDepthParams &md,
                                       bool gamma_corr = false) override;

    virtual CPT_CPU const float *render_raw(const MaxDepthParams &md,
                                            bool gamma_corr = false) override;

  private:
    void allocate_restir_buffers();
    void swap_reservoirs();
};
