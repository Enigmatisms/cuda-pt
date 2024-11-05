/**
 * @file bvh_opt.cuh
 * 
 * @author Qianyue He
 * @brief BVH optimizer from paper 
 * Fast Insertion-Based Optimization of
 * Bounding Volume Hierarchies
 * 
 * Current state: I have not yet understand the idea behind the paper
 * So, I will use a bruteforce method (with some default parameters)
 * @date 2024-11-05
 * @copyright Copyright (c) 2024
 */
#include "core/bvh.cuh"

// Get SAH cost for the BVH tree
float calculate_cost(const BVHNode* root, float traverse_cost);