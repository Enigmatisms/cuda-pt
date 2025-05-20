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
 * @author Qianyue He
 * @brief Ray tracing depth configs
 * @date  Unknown
 */
#pragma once
#include <utility>

struct MaxDepthParams {
    int max_diffuse;
    int max_specular;
    int max_tranmit;
    int max_volume;
    int max_depth;
    float min_time;
    float max_time;

    MaxDepthParams(int max_d = 4, int max_s = 2, int max_t = 8, int max_v = 3,
                   int max_total = 8)
        : max_diffuse(max_d), max_specular(max_s), max_tranmit(max_t),
          max_volume(max_v), min_time(0), max_time(0),
          max_depth(
              std::max(std::max(max_total, max_t), std::max(max_d, max_s))) {}
};
