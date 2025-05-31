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
static void calculate_cost_recursive(NodeType *node, float &sum_leaf,
                                     int &non_leaf, int &leaf) {
    if (node->lchild) { // BVH intermediate node will have both lchild, rchild,
                        // or neither
        non_leaf++;
        calculate_cost_recursive(node->lchild, sum_leaf, non_leaf, leaf);
        calculate_cost_recursive(node->rchild, sum_leaf, non_leaf, leaf);
    } else { // BVH leaf node
        leaf++;
        sum_leaf +=
            node->bound.area() * static_cast<float>(node->bound.prim_cnt());
    }
}

// Get SAH cost for the BVH tree
template <typename NodeType>
float calculate_cost(NodeType *root, float traverse_cost) {
    float sum_leaf = 0, root_area = root->bound.area();
    int num_non_leaf = 0, num_leaf = 0;
    calculate_cost_recursive(root->lchild, sum_leaf, num_non_leaf, num_leaf);
    calculate_cost_recursive(root->rchild, sum_leaf, num_non_leaf, num_leaf);
    return traverse_cost * static_cast<float>(num_non_leaf) +
           sum_leaf / root_area;
}

template float calculate_cost<BVHNode>(BVHNode *root, float traverse_cost);
template float calculate_cost<SBVHNode>(SBVHNode *root, float traverse_cost);
