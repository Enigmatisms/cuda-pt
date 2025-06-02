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

    float left_cost = calculate_SAH_recursive(
        node->lchild, cost_traverse, cost_traverse_spatial, cost_intersect);
    float right_cost = calculate_SAH_recursive(
        node->rchild, cost_traverse, cost_traverse_spatial, cost_intersect);

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
    return calculate_SAH_recursive(root, traverse_cost, spatial_traverse_cost,
                                   intersect_cost);
}

template <typename NodeType>
void level_order_traverse(const NodeType *const root, int max_level) {
    std::vector<std::pair<int, const NodeType *>> queue = {{0, root}};
    int level = 1;
    int node_cnt = 1;
    while (!queue.empty() && level <= max_level) {
        printf("Level: %d (node num: %lu)\n", level, queue.size());
        std::vector<std::pair<int, const NodeType *>> new_queue = {};
        new_queue.reserve(queue.size() * 2);

        for (auto [father, node] : queue) {
            printf("(%d)(sz: %d, lf: %d, fa: %d), ", node_cnt, node->prim_num(),
                   int(node->lchild == nullptr), father);
            if (node->lchild)
                new_queue.emplace_back(node_cnt, node->lchild);
            if (node->rchild)
                new_queue.emplace_back(node_cnt, node->rchild);
            node_cnt++;
        }
        printf("\n");
        queue = std::move(new_queue);
        level += 1;
    }
}

template float calculate_cost<BVHNode>(BVHNode *root, float traverse_cost,
                                       float spatial_traverse_cost,
                                       float intersect_cost);
template float calculate_cost<SBVHNode>(SBVHNode *root, float traverse_cost,
                                        float spatial_traverse_cost,
                                        float intersect_cost);

template void level_order_traverse<BVHNode>(const BVHNode *const root,
                                            int max_level);
template void level_order_traverse<SBVHNode>(const SBVHNode *const root,
                                             int max_level);
