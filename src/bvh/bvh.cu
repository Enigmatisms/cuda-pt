/**
 * @file bvh.cpp
 * @author Qianyue He
 * @date 2023.5 -> 2024.9
 * @brief BVH construction main logic
 * @copyright Copyright (c) 2023-2024
 */

#include <algorithm>
#include <numeric>
#include <array>
#include "bvh.cuh"

using IntPair = std::pair<int, bool>;

static constexpr int num_bins = 12;
static constexpr float traverse_cost = 0.4;
static constexpr float max_node_prim = 1;

SplitAxis BVHNode::max_extent_axis(const std::vector<BVHInfo>& bvhs, std::vector<float>& bins) const {
    Vec3 min_ctr = bvhs[base].centroid, max_ctr = bvhs[base].centroid;
    for (int i = 1; i < prim_num; i++) {
        Vec3 ctr = bvhs[base + i].centroid;
        min_ctr.minimized(ctr);
        max_ctr.maximized(ctr);
    }
    Vec3 diff = max_ctr - min_ctr;
    float max_diff = diff.x();
    int split_axis = 0;
    if (diff.y() > max_diff) {
        max_diff = diff.y();
        split_axis = 1;
    }
    if (diff.z() > max_diff) {
        max_diff = diff.z();
        split_axis = 2;
    }
    bins.resize(num_bins);
    float min_r = min_ctr[split_axis] - 0.001f, interval = (max_diff + 0.002f) / float(num_bins);
    std::transform(bins.begin(), bins.end(), bins.begin(), [min_r, interval, i = 0] (const float&) mutable {
        i++; return min_r + interval * float(i);
    });
    return SplitAxis(split_axis);
}

void index_input(
    const std::vector<ObjInfo>& objs, 
    const std::vector<bool>& sphere_flags, 
    std::vector<IntPair>& idxs, size_t num_primitives
) {
    // input follow the shape of the number of objects, for each position
    // the number of primitive / whether the primitive is sphere will be stored, the index will be object id
    size_t result_shape = objs.size();      // shape is (3, obj_num)
    idxs.reserve(num_primitives);                   // accumulate(num_ptr, num_ptr + result_shape) = num_primitives
    for (size_t i = 0; i < result_shape; i++) {
        const int prim_num = objs[i].prim_num;
        const bool sphere_flag = sphere_flags[i];
        for (int j = 0; j < prim_num; j++)
            idxs.emplace_back(static_cast<int>(i), sphere_flag);
    }
}

void create_bvh_info(
    const std::vector<Vec3>& points1,
    const std::vector<Vec3>& points2,
    const std::vector<Vec3>& points3,
    const std::vector<IntPair>& idxs, std::vector<BVHInfo>& bvh_infos) {
    bvh_infos.reserve(points1.size());
    printf("Point1 size: %lu, idx pair size: %lu\n", points1.size(), points2.size());
    for (size_t i = 0; i < points1.size(); i++) {
        const IntPair& idx_info = idxs[i];
        bvh_infos.emplace_back(points1[i], points2[i], points3[i], idx_info.first, idx_info.second);
    }
}

int recursive_bvh_SAH(BVHNode* const cur_node, std::vector<BVHInfo>& bvh_infos, int depth = 0) {
    AABB fwd_bound, bwd_bound;
    int child_prim_cnt = 0;                // this index is used for indexing variable `bins`
    const int prim_num = cur_node->prim_num, base = cur_node->base, max_pos = base + prim_num;
    float min_cost = 5e9, node_prim_cnt = float(prim_num), node_inv_area = 1. / cur_node->bound.area();

    // Step 1: decide the axis that expands the maximum extent of space
    std::vector<float> bins;        // bins: from (start_pos + interval) to end_pos
    SplitAxis max_axis = cur_node->max_extent_axis(bvh_infos, bins);
    printf("Max extend axis: %d, %d\n", max_axis, prim_num);
    if (cur_node->prim_num > 4) {   // SAH

        // Step 2: binning the space
        std::array<AxisBins, num_bins> idx_bins;
        for (int i = cur_node->base; i < max_pos; i++) {
            size_t index = std::lower_bound(bins.begin(), bins.end(), bvh_infos[i].centroid[max_axis]) - bins.begin();
            idx_bins[index].push(bvh_infos[i]);
        }

        // Step 3: forward-backward linear sweep for heuristic calculation
        std::array<int, num_bins> prim_cnts;
        std::array<float, num_bins> fwd_areas, bwd_areas;
        for (int i = 0; i < num_bins; i++) {
            fwd_bound   += idx_bins[i].bound;
            prim_cnts[i] = idx_bins[i].prim_cnt;
            fwd_areas[i] = fwd_bound.area();
            if (i > 0) {
                bwd_bound += idx_bins[num_bins - i].bound;
                bwd_areas[num_bins - 1 - i] = bwd_bound.area();
            }
        }
        std::partial_sum(prim_cnts.begin(), prim_cnts.end(), prim_cnts.begin());

        // Step 4: use the calculated area to computed the segment boundary
        int seg_bin_idx = 0;
        for (int i = 0; i < num_bins - 1; i++) {
            float cost = traverse_cost + node_inv_area * 
                (float(prim_cnts[i]) * fwd_areas[i] + (node_prim_cnt - (prim_cnts[i])) * bwd_areas[i]);
            if (cost < min_cost) {
                min_cost = cost;
                seg_bin_idx = i;
            }
        }
        // Step 5: reordering the BVH info in the vector to make the segment contiguous (partition around pivot)
        if (min_cost < node_prim_cnt) {
            std::partition(bvh_infos.begin() + base, bvh_infos.begin() + max_pos,
                [pivot = bins[seg_bin_idx], dim = max_axis](const BVHInfo& bvh) {
                    return bvh.centroid[dim] < pivot;
            });
            child_prim_cnt = prim_cnts[seg_bin_idx];
        }
        fwd_bound.clear();
        bwd_bound.clear();
        for (int i = 0; i <= seg_bin_idx; i++)       // calculate child node bound
            fwd_bound += idx_bins[i].bound;
        for (int i = num_bins - 1; i > seg_bin_idx; i--)
            bwd_bound += idx_bins[i].bound;
    } else {                                    // equal primitive number split (two nodes have identical primitives)
        int seg_idx = (base + max_pos) >> 1;
        // Step 5: reordering the BVH info in the vector to make the segment contiguous (keep around half of the bvh in lchild)
        std::nth_element(bvh_infos.begin() + base, bvh_infos.begin() + seg_idx, bvh_infos.begin() + max_pos,
            [dim = max_axis] (const BVHInfo& bvh1, const BVHInfo& bvh2) {
                return bvh1.centroid[dim] < bvh2.centroid[dim];
            }
        );
        for (int i = base; i < seg_idx; i++)    // calculate child node bound
            fwd_bound += bvh_infos[i].bound;
        for (int i = seg_idx; i < max_pos; i++)
            bwd_bound += bvh_infos[i].bound;
        child_prim_cnt = seg_idx - base;        // bvh[seg_idx] will be in rchild
        float split_cost = traverse_cost + node_inv_area * 
                (fwd_bound.area() * child_prim_cnt + bwd_bound.area() * (node_prim_cnt - child_prim_cnt));
        if (split_cost >= node_prim_cnt)
            child_prim_cnt = 0;
    }
    if (child_prim_cnt > 0) {             // cost of splitting is less than making this node a leaf node
        // Step 5: split the node and initialize the children
        cur_node->lchild = new BVHNode(base, child_prim_cnt);
        cur_node->rchild = new BVHNode(base + child_prim_cnt, prim_num - child_prim_cnt);

        cur_node->lchild->bound = fwd_bound;
        cur_node->rchild->bound = bwd_bound;
        cur_node->axis = max_axis;
        // Step 7: start recursive splitting for the children
        int node_num = 1;
        // if (depth == 0) {
        //     int local_node_n1 = 0, local_node_n2 = 0;
        //     #pragma omp parallel sections       // parallel SAH-BVH
        //     {
        //         #pragma omp section 
        //         {
        //             if (cur_node->lchild->prim_num > max_node_prim)
        //                 local_node_n1 = recursive_bvh_SAH(cur_node->lchild, bvh_infos, depth + 1);
        //             else local_node_n1 = 1;
        //         }
        //         #pragma omp section 
        //         {
        //             if (cur_node->rchild->prim_num > max_node_prim)
        //                 local_node_n2 = recursive_bvh_SAH(cur_node->rchild, bvh_infos, depth + 1);
        //             else local_node_n2 = 1;
        //         }
        //     }
        //     node_num = local_node_n1 + local_node_n2 + 1;
        // } else {
            if (cur_node->lchild->prim_num > max_node_prim)
                node_num += recursive_bvh_SAH(cur_node->lchild, bvh_infos, depth + 1);
            else node_num ++;
            if (cur_node->rchild->prim_num > max_node_prim)
                node_num += recursive_bvh_SAH(cur_node->rchild, bvh_infos, depth + 1);
            else node_num ++;
        // }
        return node_num;
    } else {
        // This is a leaf node, yet this is the only way that a leaf node contains more than one primitive
        cur_node->axis = AXIS_NONE;
        return 1;
    }
}

static BVHNode* bvh_root_start(const Vec3& world_min, const Vec3& world_max, int& node_num, std::vector<BVHInfo>& bvh_infos) {
    // Build BVH tree root node and start recursive tree construction
    printf("World min: ");
    print_vec3(world_min);
    printf("World max: ");
    print_vec3(world_max);
    BVHNode* root_node = new BVHNode(0, bvh_infos.size());
    root_node->bound.mini = world_min;
    root_node->bound.maxi = world_max;
    node_num = recursive_bvh_SAH(root_node, bvh_infos);
    return root_node;
}

// This is the final function call for `bvh_build`
static int recursive_linearize(BVHNode* cur_node, std::vector<LinearNode>& lin_nodes) {
    // BVH tree should be linearized to better traverse and fit in the system memory
    // The linearized BVH tree should contain: bound, base, prim_cnt, rchild_offset, total_offset (to skip the entire node)
    // Note that if rchild_offset is -1, then the node is leaf. Leaf node points to primitive array
    // which is already sorted during BVH construction, containing primitive_id and obj_id for true intersection
    // Note that lin_nodes has been reserved
    size_t current_size = lin_nodes.size();
    lin_nodes.emplace_back(cur_node);
    if (cur_node->lchild != nullptr) {
        // TODO: parallel linearize
        int lnodes = recursive_linearize(cur_node->lchild, lin_nodes);
        lnodes += recursive_linearize(cur_node->rchild, lin_nodes);
        lin_nodes[current_size].all_offset = lnodes + 1;
        return lnodes + 1;                      // include the cur_node                       
    } else {
        printf("Lin node size: %lu. Recursive End.\n", lin_nodes.size());
        lin_nodes.back().all_offset = 1;        // to skip the current sub-tree, index should just add 1
        return 1;
    }
}

// Try to use two threads to build the BVH
void bvh_build(
    const std::vector<Vec3>& points1,
    const std::vector<Vec3>& points2,
    const std::vector<Vec3>& points3,
    const std::vector<ObjInfo>& objects,
    const std::vector<bool>& sphere_flags,
    const Vec3& world_min, const Vec3& world_max,
    std::vector<LinearBVH>& lin_bvhs, std::vector<LinearNode>& lin_nodes
) {
    std::vector<IntPair> idx_prs;
    std::vector<BVHInfo> bvh_infos;
    int node_num = 0, num_prims_all = points1.size();
    index_input(objects, sphere_flags, idx_prs, num_prims_all);
    create_bvh_info(points1, points2, points3, idx_prs, bvh_infos);
    BVHNode* root_node = bvh_root_start(world_min, world_max, node_num, bvh_infos);
    recursive_linearize(root_node, lin_nodes);
    lin_bvhs.reserve(bvh_infos.size());
    for (const BVHInfo& bvh: bvh_infos) {
        lin_bvhs.emplace_back(bvh);
    }
    delete root_node;
}