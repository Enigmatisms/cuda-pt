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
 * @brief Some general constants
 * @date:   2024.5.12
 */

#pragma once

constexpr float EPSILON = 1e-3f;
constexpr float THP_EPS = 1e-4f;
constexpr float AABB_EPS = 1e-5f;
constexpr float MAX_DIST = 1e7;
constexpr float ENVMAP_DIST = 5e3;
constexpr float AABB_INVALID_DIST = 1e5;
constexpr float SCALING_EPS = 1.05f;

constexpr float M_Pi = 3.1415926535897f;
constexpr float M_2Pi = M_Pi * 2;
constexpr float M_1_Pi = 1.f / M_Pi;
constexpr float DEG2RAD = M_Pi / 180.f;

constexpr int INVALID_OBJ = 0xffffffff;
static constexpr int OCC_BLOCK_PER_SM = 12; // calculated by profiling
