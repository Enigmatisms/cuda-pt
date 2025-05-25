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
 * @brief BVH construction main logic
 * @date 2023.5 -> 2024.9
 */

#include "core/bvh_opt.cuh"
#include "core/bvh_spatial.cuh"
#include <algorithm>
#include <array>
#include <numeric>
#include <unordered_set>

static constexpr int num_bins = 16;
static constexpr int no_div_threshold = 2;
static constexpr int sah_split_threshold = 8;
// A cluster with all the primitive centroid within a small range [less than
// 1e-3] is ill-posed. If there is more than 64 primitives, the primitives will
// be discarded
static constexpr float traverse_cost = 0.2f;
static constexpr int unordered_threshold = 512;

static float bvh_overlap_w = 1.f;
static int max_depth = 0;

SplitAxis SBVHNode::max_extent_axis(const std::vector<BVHInfo> &bvhs,
                                    float &min_r, float &interval) const {

    Vec3 min_ctr = Vec3(std::numeric_limits<float>::max()),
         max_ctr = Vec3(std::numeric_limits<float>::min());

    for (int bvh_id : prims) {
        Vec3 ctr = bvhs[bvh_id].centroid;
        min_ctr.minimized(ctr);
        max_ctr.maximized(ctr);
    }

    Vec3 diff = max_ctr - min_ctr;
    float max_diff = diff.x();
    min_r = min_ctr[0] - 1e-5;
    int split_axis = 0;
    if (diff.y() > max_diff) {
        max_diff = diff.y();
        split_axis = 1;
        min_r = min_ctr[1] - 1e-5;
    }
    if (diff.z() > max_diff) {
        max_diff = diff.z();
        split_axis = 2;
        min_r = min_ctr[2] - 1e-5;
    }
    if (diff.max_elem() < 1e-3) {
        return SplitAxis::AXIS_NONE;
    }
    interval = (max_diff + 2e-5f) / float(num_bins);
    return SplitAxis(split_axis);
}

template <int N>
void SpatialSplitter<N>::update(const Vec3 &v1, const Vec3 &v2, const Vec3 &v3,
                                int prim_id) {
    // 1. sort the points according to the position on the split axis
    // we won't have degenerate triangles here.
    float p1_v = v1[axis], p2_v = v2[axis], p3_v = v3[axis];
    Vec3 p1, p2, p3;
    if (p1_v > p2_v)
        std::swap(v1, v2);
    // now v1[axis] <= v2[axis] always holds,
    // therefore, p1 can never be v2
    if (p3_v < p1_v) {
        p1 = v3;
        p2 = v1;
        p3 = v2;
    } else {
        p1 = v1;
        if (p2_v < p3_v) {
            p2 = v2;
            p3 = v3;
        } else {
            p2 = v3;
            p3 = v2;
        }
    }
    // After sorting, p1, p2, p3 should have increasing split axis coord
    // convert the abs position to direction and normalize
    Vec3 dir1 = p2 - p1;
    Vec3 dir2 = p3 - p1;
    dir1 *= 1.f / dir1[axis];
    dir2 *= 1.f / dir2[axis];

    // 2. get bin ID of p1, p2 and p3 and update the ID record
    int v1_id = get_bin_id(p1);
    int v2_id = get_bin_id(p2);
    int v3_id = get_bin_id(p3);
    enter_tris[v1_id].push_back(prim_id);
    exit_tris[v3_id].push_back(prim_id);

    float d2bin_start = s_pos + interval * static_cast<float>(v1_id) - p1[axis];
    Vec3 end_p1 = p1 + d2bin_start * dir1, end_p2 = p1 + d2bin_start * dir2;
    for (int id = v1_id; id <= v3_id; id++) {
        AABB &aabb = bounds[id];

        if (id != v1_id) {
            aabb.extend(end_p1);
            aabb.extend(end_p2);
        } else {
            aabb.extend(p1);
        }

        if (id == v2_id) {
            aabb.extend(p2);
            // reset the direction and normalize
            dir1 = p3 - p2;
            dir1 *= 1.f / dir1[axis];
            // reset end point 1
            end_p1 = p2 + dir1 * (s_pos + interval * static_cast<float>(v2_id) -
                                  p2[axis]);
        }

        if (id != v3_id) {
            end_p1 += interval * dir1;
            end_p2 += interval * dir2;
            aabb.extend(end_p1);
            aabb.extend(end_p2);
        } else {
            aabb.extend(p3);
        }
    }
}

int recursive_sbvh_SAH(SBVHNode *const cur_node,
                       std::vector<BVHInfo> &bvh_infos, int depth = 0,
                       int max_prim_node = 16) {
    AABB fwd_bound(1e5f, -1e5f, 0, 0), bwd_bound(1e5f, -1e5f, 0, 0);
    int child_prim_cnt = 0; // this index is used for indexing variable `bins`
    const int prim_num = cur_node->prim_num(), base = cur_node->base(),
              max_pos = base + prim_num;
    float min_cost = 5e9f, node_prim_cnt = float(prim_num);

    // Step 1: decide the axis that expands the maximum extent of space
    float min_range = 0, interval = 0;
    SplitAxis max_axis =
        cur_node->max_extent_axis(bvh_infos, min_range, interval);

    if (max_axis != SplitAxis::AXIS_NONE &&
        prim_num > sah_split_threshold) { // SAH

        // Step 2: binning all the triangles in the range, calculate exact AABB
        // for each bin

        SpatialSplitter<num_bins> spatial_split(
            min_range, min_range + interval * static_cast<float>(num_bins),
            max_axis);
        // TODO(heqianyue): this function is not yet implemented
        spatial_split(cur_node, bvh_infos);

        // Step 3: forward-backward linear sweep for heuristic calculation
        std::array<int, num_bins> prim_cnts;
        std::array<float, num_bins> fwd_areas, bwd_areas;
        prim_cnts.fill(0);
        fwd_areas.fill(0);
        bwd_areas.fill(0);
        for (int i = 0; i < num_bins; i++) {
            fwd_bound += spatial_split[i];
            prim_cnts[i] = spatial_split.num_entering(i);
            if (i > 0)
                prim_cnts[i] += prim_cnts[i - 1];
            fwd_areas[i] = fwd_bound.area();
            if (i > 0) {
                bwd_bound += spatial_split[num_bins - i];
                bwd_areas[num_bins - 1 - i] = bwd_bound.area();
            }
        }
        cur_node->bound.mini = fwd_bound.mini;
        cur_node->bound.maxi = fwd_bound.maxi;
        float node_inv_area = 1. / cur_node->bound.area();

        // Step 4: use SAH to calculate the best spatial split
        int seg_bin_idx = 0;
        for (int i = 0; i < num_bins - 1; i++) {
            float cost =
                traverse_cost +
                node_inv_area *
                    (float(prim_cnts[i]) * fwd_areas[i] +
                     (node_prim_cnt - float(prim_cnts[i])) * bwd_areas[i]);
            if (cost < min_cost) {
                min_cost = cost;
                seg_bin_idx = i;
            }
        }
        // Step 5: use entering and exiting records to get the primitives
        // inside the split range (for lchild and rchild)
        if (min_cost < node_prim_cnt || prim_num > max_prim_node) {
            // the primitives in the left node: entering on the left
            // the primitives in the right node: not exiting on the left
            // TODO(heqianyue): check the following
            child_prim_cnt = prim_cnts[seg_bin_idx];
            std::vector<int> left_prims, right_prims;
            left_prims.reserve(child_prim_cnt);
            right_prims.reserve(prim_num / 2);
            std::unordered_set<int> exit_from_left;
            for (int i = 0; i <= seg_bin_idx; i++) {
                left_prims.insert(left_prims.begin(),
                                  spatial_split.enter_tris[i].begin(),
                                  spatial_split.enter_tris[i].end());
                for (int v : spatial_split.exit_tris[i]) {
                    exit_from_left.emplace(v);
                }
            }
            for (int prim_id : cur_node->prims) {
                if (exit_from_left.count(prim_id))
                    continue;
                right_prims.push_back(prim_id);
            }
        }

        fwd_bound.clear();
        bwd_bound.clear();
        for (int i = 0; i <= seg_bin_idx; i++) // calculate child node bound
            fwd_bound += spatial_split[i];
        for (int i = seg_bin_idx + 1; i < num_bins; i++)
            bwd_bound += spatial_split[i];
    } else { // equal primitive number split (two nodes have identical
             // primitives)
        int seg_idx = (base + max_pos) >> 1;
        // Step 5: reordering the BVH info in the vector to make the segment
        // contiguous (keep around half of the bvh in lchild)
        if (max_axis != SplitAxis::AXIS_NONE) {
            std::nth_element(
                bvh_infos.begin() + base, bvh_infos.begin() + seg_idx,
                bvh_infos.begin() + max_pos,
                [dim = max_axis](const BVHInfo &bvh1, const BVHInfo &bvh2) {
                    return bvh1.centroid[dim] < bvh2.centroid[dim];
                });
        }
        for (int i = base; i < seg_idx; i++) // calculate child node bound
            fwd_bound += bvh_infos[i].bound;
        for (int i = seg_idx; i < max_pos; i++)
            bwd_bound += bvh_infos[i].bound;
        cur_node->bound += fwd_bound;
        cur_node->bound += bwd_bound;
        child_prim_cnt = seg_idx - base; // bvh[seg_idx] will be in rchild
        float intrsct_a = fwd_bound.intersection_area(bwd_bound);
        float split_cost =
            traverse_cost +
            (1.f / cur_node->bound.area()) *
                (intrsct_a * std::max(bvh_overlap_w - 0.5f, 0.f) *
                     node_prim_cnt +
                 fwd_bound.area() * float(child_prim_cnt) +
                 bwd_bound.area() * (node_prim_cnt - float(child_prim_cnt)));
        if (split_cost >= node_prim_cnt && prim_num < max_prim_node)
            child_prim_cnt = 0;
    }

    if (child_prim_cnt >
        0) { // cost of splitting is less than making this node a leaf node
        // Step 5: split the node and initialize the children
        cur_node->lchild = new BVHNode(base, child_prim_cnt);
        cur_node->rchild =
            new BVHNode(base + child_prim_cnt, prim_num - child_prim_cnt);

        cur_node->lchild->bound.mini = fwd_bound.mini;
        cur_node->lchild->bound.maxi = fwd_bound.maxi;
        cur_node->rchild->bound.mini = bwd_bound.mini;
        cur_node->rchild->bound.maxi = bwd_bound.maxi;
        cur_node->axis = max_axis;
        // Step 7: start recursive splitting for the children
        int node_num = 1;
        if (cur_node->lchild->prim_num() > no_div_threshold)
            node_num += recursive_bvh_SAH(cur_node->lchild, bvh_infos,
                                          depth + 1, max_prim_node);
        else {
            max_depth = std::max(depth + 1, max_depth);
            node_num++;
        }
        if (cur_node->rchild->prim_num() > no_div_threshold)
            node_num += recursive_bvh_SAH(cur_node->rchild, bvh_infos,
                                          depth + 1, max_prim_node);
        else {
            max_depth = std::max(depth + 1, max_depth);
            node_num++;
        }
        return node_num;
    } else {
        // This is a leaf node, yet this is the only way that a leaf node
        // contains more than one primitive
        cur_node->axis = AXIS_NONE;
        max_depth = std::max(depth, max_depth);
        return 1;
    }
}
