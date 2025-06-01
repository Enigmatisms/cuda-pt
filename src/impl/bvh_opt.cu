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
 * @brief BVH optimization
 * @date Unknown
 */
#include "core/bvh_opt.cuh"

template <typename NodeType>
float calculate_SAH_recursive(NodeType *node, float cost_traverse,
                              float cost_traverse_spatial = 0.4f,
                              float cost_intersect = 1.f) {
    if (node == nullptr) {
        return 0.0;
    }

    if (node->lchild == nullptr || node->rchild == nullptr) {
        return cost_intersect * static_cast<float>(node->prim_num());
    }

    float node_area = node->bound.area();

    float left_area = node->lchild->bound.area();
    float right_area = node->rchild->bound.area();

    float left_cost =
        calculate_SAH_recursive(node->lchild, cost_traverse, cost_intersect);
    float right_cost =
        calculate_SAH_recursive(node->rchild, cost_traverse, cost_intersect);

    float inv_area = (node_area > 0) ? 1.0 / node_area : 0.0;
    float left_ratio = left_area * inv_area;
    float right_ratio = right_area * inv_area;
    if constexpr (std::is_same_v<std::decay_t<NodeType>, SBVHNode>) {
        return (node->axis > SplitAxis::AXIS_NONE ? cost_traverse_spatial
                                                  : cost_traverse) +
               left_ratio * left_cost + right_ratio * right_cost;
    } else {
        return cost_traverse + left_ratio * left_cost +
               right_ratio * right_cost;
    }
}

// Get SAH cost for the BVH tree
template <typename NodeType>
float calculate_cost(NodeType *root, float traverse_cost,
                     float spatial_traverse_cost, float intersect_cost) {
    return calculate_SAH_recursive(root, traverse_cost);
}

template float calculate_cost<BVHNode>(BVHNode *root, float traverse_cost,
                                       float spatial_traverse_cost,
                                       float intersect_cost);
template float calculate_cost<SBVHNode>(SBVHNode *root, float traverse_cost,
                                        float spatial_traverse_cost,
                                        float intersect_cost);
