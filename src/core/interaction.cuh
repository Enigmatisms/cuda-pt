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
 * Ray intersection (or scattering event) interaction point
 * @author: Qianyue He
 * @date:   4.29.2024
 */
#pragma once
#include "core/vec2_half.cuh"

// Now, this data type can fit in a 16 Byte float4 (128 bit L/S is possible)
class Interaction {
  public:
    Vec3 shading_norm; // size: 3 floats
    Vec2Half uv_coord; // size: 1 float

    CPT_CPU_GPU Interaction() {}

    template <typename Vec3Type, typename Vec2Type>
    CPT_CPU_GPU Interaction(Vec3Type &&_n, Vec2Type &&_uv)
        : shading_norm(std::forward<Vec3Type>(_n)),
          uv_coord(std::forward<Vec2Type>(_uv)) {
        static_assert(
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<Vec3Type>>,
                           Vec3> &&
                (std::is_same_v<
                     std::remove_cv_t<std::remove_reference_t<Vec2Type>>,
                     Vec2Half> ||
                 std::is_same_v<
                     std::remove_cv_t<std::remove_reference_t<Vec2Type>>,
                     Vec2>),
            "Input type check failed");
    }
};
