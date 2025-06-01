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
 * @brief BVH optimizer from paper
 * Fast Insertion-Based Optimization of
 * Bounding Volume Hierarchies
 *
 * Current state: I have not yet understand the idea behind the paper
 * So, I will use a bruteforce method (with some default parameters)
 * @date 2024.11.05
 */
#pragma once
#include "core/bvh.cuh"
#include "core/bvh_spatial.cuh"

// Get SAH cost for the BVH tree
template <typename NodeType>
float calculate_cost(NodeType *root, float traverse_cost,
                     float spatial_traverse_cost = 0.4f,
                     float intersect_cost = 1.f);
