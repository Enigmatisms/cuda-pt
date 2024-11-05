#include "core/bvh_opt.cuh"

static void calculate_cost_recursive(const BVHNode* node, float& sum_intr, float& sum_leaf, int& non_leaf, int& leaf) {
    if (node->lchild) {     // BVH intermediate node will have both lchild, rchild, or neither
        non_leaf ++;
        sum_intr += node->bound.area();
        calculate_cost_recursive(node->lchild, sum_intr, sum_leaf, non_leaf, leaf);
        calculate_cost_recursive(node->rchild, sum_intr, sum_leaf, non_leaf, leaf);
    } else {                // BVH leaf node
        leaf ++;
        sum_leaf += node->bound.area() * static_cast<float>(node->bound.prim_cnt());
    }
}

// Get SAH cost for the BVH tree
float calculate_cost(const BVHNode* root, float traverse_cost) {
	float sum_intr = 0, sum_leaf = 0, root_area = root->bound.area();
    int num_non_leaf = 0, num_leaf = 0;
    calculate_cost_recursive(root->lchild, sum_intr, sum_leaf, num_non_leaf, num_leaf);
    calculate_cost_recursive(root->rchild, sum_intr, sum_leaf, num_non_leaf, num_leaf);
    return (traverse_cost * sum_intr + sum_leaf) / root_area;
}