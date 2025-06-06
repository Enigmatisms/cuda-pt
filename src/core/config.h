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
 * @brief Rendering configuration
 * @date:   2024.5.24
 */

#pragma once
#include "core/max_depth.h"
#include <string>
#include <tinyxml2.h>

class RenderingConfig {
  public:
    struct BVHConfig {
        int max_node_num = 16;
        int cache_level = 4;
        float bvh_overlap_w = 0.5; // [0.5, +inf), can not be less than 0.5,
                                   // otherwise SAH will be downweighted
        bool use_sbvh = false;
        bool use_ref_unsplit = true;
    } bvh;

    int spp = 256;
    int width = 1024;
    int height = 1024;
    int spec_constraint = 0;
    bool gamma_correction = true;
    bool bidirectional = false;
    float caustic_scaling = 1.f;

    MaxDepthParams md;

    static RenderingConfig from_xml(const tinyxml2::XMLElement *acc_node,
                                    const tinyxml2::XMLElement *render_node,
                                    const tinyxml2::XMLElement *sensor_node);
};
