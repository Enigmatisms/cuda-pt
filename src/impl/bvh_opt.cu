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
float calculate_SAH_recursive(const NodeType *const node, float cost_traverse,
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
float calculate_cost(const NodeType *const root, float traverse_cost,
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

struct SubtreeStats {
    int height;
    int prim_count;
};

struct TreeMetrics {
    float avg_tree_hdiff = 0.0;      // average tree height difference (abs)
    float avg_prim_imbalance = 0.0;  // imbalance of primitives
    float avg_leaf_primitives = 0.0; // average primitives in a leaf
    float avg_overlap_factor = 0.0;
    float avg_node_intersect_factor = 0.0;
    float avg_spatial_split_overlap = 0.0;
    float min_spatial_split_overlap = 114514.1919810f;
    float max_spatial_split_overlap = 0.0;
    int min_leaf_primitives = INT_MAX; // minimum primitives in a leaf
    int max_leaf_primitives = 0;       // maximum primitives in a leaf
    int internal_nodes = 0;            // number of internal nodes
    int leaf_nodes = 0;                // number of leaf nodes
    int spatial_split_nodes = 0;       // number of nodes that use spatial split
    int bad_nodes =
        0; // number of nodes whose child nodes have greater surface area
};

#define IS_SBVH_NODE(_NodeType)                                                \
    std::is_same_v<std::decay_t<_NodeType>, SBVHNode>
template <typename NodeType>
SubtreeStats compute_tree_metrics(const NodeType *const node,
                                  TreeMetrics &metrics) {
    if (!node)
        return {-1, 0};

    // process leaf
    if (!node->lchild && !node->rchild) {
        metrics.leaf_nodes++;
        metrics.avg_leaf_primitives += node->prim_num();
        if (node->prim_num() < metrics.min_leaf_primitives)
            metrics.min_leaf_primitives = node->prim_num();
        if (node->prim_num() > metrics.max_leaf_primitives)
            metrics.max_leaf_primitives = node->prim_num();
        return {0, node->prim_num()};
    }

    // process non-leaf
    metrics.internal_nodes++;
    auto left_stats = compute_tree_metrics(node->lchild, metrics);
    auto right_stats = compute_tree_metrics(node->rchild, metrics);
    float intr_area =
        node->lchild->bound.intersection_area(node->rchild->bound);
    float curr_area = node->bound.area(),
          lchild_area = node->lchild->bound.area(),
          rchild_area = node->rchild->bound.area();
    AABB lbound = node->lchild->bound;
    lbound ^= node->rchild->bound;
    int axis = node->axis > SplitAxis::AXIS_NONE
                   ? node->axis - SplitAxis::AXIS_S_X
                   : node->axis;
    if (lbound.range()[axis] < 5e-5f)
        intr_area = 0;
    metrics.avg_overlap_factor += intr_area / curr_area;
    metrics.avg_node_intersect_factor +=
        (lchild_area + rchild_area) / curr_area;
    metrics.bad_nodes += (lchild_area > curr_area) | (rchild_area > curr_area);

    if constexpr (IS_SBVH_NODE(NodeType)) {
        if (node->axis > SplitAxis::AXIS_NONE) {
            float local_overlap = intr_area / curr_area;
            metrics.max_spatial_split_overlap =
                std::max(local_overlap, metrics.max_spatial_split_overlap);
            metrics.min_spatial_split_overlap =
                std::min(local_overlap, metrics.min_spatial_split_overlap);
            metrics.avg_spatial_split_overlap += local_overlap;
            metrics.spatial_split_nodes++;
        }
    }

    // update the metric of the current node
    int height = std::max(left_stats.height, right_stats.height) + 1;
    int prim_count = left_stats.prim_count + right_stats.prim_count;

    metrics.avg_tree_hdiff += std::abs(left_stats.height - right_stats.height);

    // calculate imbalance factor
    float total_prims = left_stats.prim_count + right_stats.prim_count;
    if (total_prims > 0) {
        double imbalance =
            std::abs(left_stats.prim_count - right_stats.prim_count) /
            total_prims;
        metrics.avg_prim_imbalance += imbalance;
    }

    return {height, prim_count};
}

template <typename NodeType>
void calculate_tree_metrics(const NodeType *const root) {
    TreeMetrics metrics;
    if (!root) {
        std::cout << "\n[Accelerator] Empty Tree. Exiting... \n";
        return;
    }

    compute_tree_metrics(root, metrics);

    if (metrics.internal_nodes > 0) {
        metrics.avg_tree_hdiff /= metrics.internal_nodes;
        metrics.avg_prim_imbalance /= metrics.internal_nodes;
        metrics.avg_overlap_factor /= metrics.internal_nodes;
        metrics.avg_node_intersect_factor /= metrics.internal_nodes;
    }

    if (metrics.leaf_nodes > 0) {
        metrics.avg_leaf_primitives /= metrics.leaf_nodes;
    }

    std::cout << "\n[Accelerator] Tree Statistics: \n";
    std::cout << "\t Avg Tree Height difference(↓):\t" << metrics.avg_tree_hdiff
              << "\n";
    std::cout << "\t Avg Primitive Imbalance(↓):\t"
              << metrics.avg_prim_imbalance << "\n";
    std::cout << "\t Avg Leaf Primitive Cnt(↓):\t"
              << metrics.avg_leaf_primitives << "\n";
    std::cout << "\t Avg AABB Overlap Factor(↓):\t"
              << metrics.avg_overlap_factor << "\n";
    std::cout << "\t Avg Intersection Factor(↓):\t"
              << metrics.avg_node_intersect_factor << "\n";
    if constexpr (IS_SBVH_NODE(NodeType)) {
        std::cout << "\t Avg Spatial Split Overlap(↓):\t"
                  << metrics.avg_spatial_split_overlap /
                         metrics.spatial_split_nodes
                  << "\n";
        std::cout << "\t Min Spatial Split Overlap(↓):\t"
                  << metrics.min_spatial_split_overlap << "\n";
        std::cout << "\t Max Spatial Split Overlap(↓):\t"
                  << metrics.max_spatial_split_overlap << "\n";
        std::cout << "\t Spatial Split Node Cnt:\t"
                  << metrics.spatial_split_nodes << "\n";
    }
    std::cout << "\t Min Leaf Primitive Cnt:\t" << metrics.min_leaf_primitives
              << "\n";
    std::cout << "\t Max Leaf Primitive Cnt:\t" << metrics.max_leaf_primitives
              << "\n";
    std::cout << "\t Internal Node Count:\t\t" << metrics.internal_nodes
              << "\n";
    std::cout << "\t Bad Node Cnt:\t" << metrics.spatial_split_nodes << "\n";
    std::cout << "\t Leaf Node Count:\t\t" << metrics.leaf_nodes << "\n\n";
    std::cout << "\t Total Node Count:\t\t"
              << metrics.leaf_nodes + metrics.internal_nodes << "\n\n";
}
#undef IS_SBVH_NODE

template float calculate_cost<BVHNode>(const BVHNode *const root,
                                       float traverse_cost,
                                       float spatial_traverse_cost,
                                       float intersect_cost);
template float calculate_cost<SBVHNode>(const SBVHNode *const root,
                                        float traverse_cost,
                                        float spatial_traverse_cost,
                                        float intersect_cost);

template void level_order_traverse<BVHNode>(const BVHNode *const root,
                                            int max_level);
template void level_order_traverse<SBVHNode>(const SBVHNode *const root,
                                             int max_level);

template void calculate_tree_metrics<BVHNode>(const BVHNode *const root);
template void calculate_tree_metrics<SBVHNode>(const SBVHNode *const root);
