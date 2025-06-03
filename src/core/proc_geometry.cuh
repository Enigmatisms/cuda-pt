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
 * @brief Process Geometries
 * @author Qianyue He
 * @date 2025.6.3
 */
#pragma once

#include "core/aabb.cuh"

CPT_CPU std::vector<Vec3> clip_polygon(std::vector<Vec3> &&polygon, int axis,
                                       float boundary_val,
                                       bool is_min_boundary);

CPT_CPU std::vector<Vec3> aabb_triangle_clipping(const AABB &aabb,
                                                 const Vec3 &p1, const Vec3 &p2,
                                                 const Vec3 &p3);

CPT_CPU std::vector<Vec3> aabb_triangle_clipping(const AABB &aabb,
                                                 std::vector<Vec3> &&polygon);
