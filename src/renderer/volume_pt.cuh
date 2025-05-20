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

#pragma once
/**
 * @file volume_pt.cuh
 * @author Qianyue He
 * @brief Volume Path Tracer class definition
 * @version 0.1
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 */

#include "core/medium.cuh"
#include "renderer/path_tracer.cuh"

class VolumePathTracer : public PathTracer {
  protected:
    int cam_vol_id;
    Medium **media;

  public:
    VolumePathTracer(const Scene &scene);

    ~VolumePathTracer();

    virtual CPT_CPU std::vector<uint8_t>
    render(const MaxDepthParams &md, int num_iter = 64,
           bool gamma_correction = true) override;

    virtual CPT_CPU void render_online(const MaxDepthParams &md,
                                       bool gamma_corr = false) override;

    virtual CPT_CPU const float *render_raw(const MaxDepthParams &md,
                                            bool gamma_corr = false) override;
};
