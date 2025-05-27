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
 * @brief Spatial BVH construction main logic
 * @date 2025.5.25
 */

#include "core/bvh_opt.cuh"
#include "core/bvh_spatial.cuh"
#include <numeric>

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

inline int object_index_packing(int obj_med_idx, int obj_id, bool is_sphere) {
    // take the lower 20 bits and shift up 20bits
    int truncated = (obj_med_idx & 0x00000fff) << 20;
    return (static_cast<int>(is_sphere) << 31) + truncated +
           (obj_id & 0x000fffff);
}

// TODO(heqianyue): note that we currently don't support
// sphere primitive. Support it would be straightforward:
// overload the 'update' function for spheres
int recursive_sbvh_SAH(const std::vector<Vec3> &points1,
                       const std::vector<Vec3> &points2,
                       const std::vector<Vec3> &points3,
                       std::vector<int> &flattened_idxs,
                       SBVHNode *const cur_node,
                       std::vector<BVHInfo> &bvh_infos, int depth = 0,
                       int max_prim_node = 16) {
    auto process_leaf = [&]() {
        // leaf node processing function
        cur_node->axis = AXIS_NONE;
        cur_node->base() = static_cast<int>(flattened_idxs.size());
        cur_node->prim_num() = static_cast<int>(cur_node->prims.size());
        max_depth = std::max(depth, max_depth);
        for (int prim_id : cur_node->prims) {
            flattened_idxs.push_back(prim_id);
        }
        return 1;
    };

    if (cur_node->size() <= no_div_threshold) {
        return process_leaf();
    }
    AABB fwd_bound(1e5f, -1e5f, 0, 0), bwd_bound(1e5f, -1e5f, 0, 0);
    int child_prim_cnt = 0; // this index is used for indexing variable `bins`
    const int prim_num = cur_node->size();
    float min_cost = 5e9f, node_prim_cnt = float(prim_num);

    // Step 1: decide the axis that expands the maximum extent of space
    float min_range = 0, interval = 0;
    SplitAxis max_axis =
        cur_node->max_extent_axis(bvh_infos, min_range, interval);

    std::vector<int> lchild_idxs, rchild_idxs;
    lchild_idxs.reserve(prim_num / 2);
    rchild_idxs.reserve(prim_num / 2);

    if (max_axis != SplitAxis::AXIS_NONE &&
        prim_num > sah_split_threshold) { // SAH
        // Step 2: binning the space
        std::array<AxisBins, num_bins> idx_bins;
        for (int bvh_id : cur_node->prims) {
            int index = std::min(
                (int)floorf((bvh_infos[bvh_id].centroid[max_axis] - min_range) /
                            interval),
                num_bins - 1);
            idx_bins[index].push(bvh_infos[bvh_id]);
        }

        // Step 3: forward-backward linear sweep for heuristic calculation
        std::array<int, num_bins> prim_cnts;
        std::array<float, num_bins> fwd_areas, bwd_areas;

        prim_cnts.fill(0);
        fwd_areas.fill(0);
        bwd_areas.fill(0);
        for (int i = 0; i < num_bins; i++) {
            fwd_bound += idx_bins[i].bound;
            prim_cnts[i] = idx_bins[i].prim_cnt;
            fwd_areas[i] = fwd_bound.area();
            if (i > 0) {
                bwd_bound += idx_bins[num_bins - i].bound;
                bwd_areas[num_bins - 1 - i] = bwd_bound.area();
            }
        }
        cur_node->bound.mini = fwd_bound.mini;
        cur_node->bound.maxi = fwd_bound.maxi;
        float node_inv_area = 1. / cur_node->bound.area();
        std::partial_sum(prim_cnts.begin(), prim_cnts.end(), prim_cnts.begin());

        // Step 4: use the calculated area to computed the segment boundary, for
        // SBVH there is no need using spatial overlap penalty for BVH
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

        if (false /*Some unknown criteria I didn't come up with yet*/) {
            // if the crieria are met, we calculate the SBVH split cost
            SpatialSplitter<num_bins> ssp(
                min_range, min_range + interval * static_cast<float>(num_bins),
                max_axis);

            ssp.update_bins(points1, points2, points3, cur_node);

            int sbvh_seg_idx = 0;
            float sbvh_cost =
                ssp.eval_spatial_split(cur_node, sbvh_seg_idx, traverse_cost);
            if (sbvh_cost < min_cost &&
                (sbvh_cost < node_prim_cnt ||
                 prim_num > max_prim_node)) { // Spatial split should be applied
                min_cost = sbvh_cost;
                ssp.apply_spatial_split(cur_node, lchild_idxs, rchild_idxs,
                                        sbvh_seg_idx);
                fwd_bound = ssp.partial_sum<false>(sbvh_seg_idx);
                bwd_bound = ssp.partial_sum<true>(sbvh_seg_idx);
            }
        }

        // 1. SBVH is not applied ; 2. when the cost of splitting is lower or 3.
        // when there are more primitives than allowed
        if (lchild_idxs.empty() && (min_cost < node_prim_cnt ||
                                    prim_num > max_prim_node)) { // object split
            // We cannot partition here, since partition will change the index
            // of the BVH
            float pivot = min_range + interval * float(seg_bin_idx + 1);
            for (int bvh_id : cur_node->prims) {
                const BVHInfo &bvh = bvh_infos[bvh_id];
                if (bvh.centroid[max_axis] < pivot) {
                    lchild_idxs.push_back(bvh_id);
                } else {
                    rchild_idxs.push_back(bvh_id);
                }
            }
            // TODO(heqianyue): child_prim_cnt might be able to be removed
            child_prim_cnt = prim_cnts[seg_bin_idx];
            fwd_bound.clear();
            bwd_bound.clear();
            for (int i = 0; i <= seg_bin_idx; i++) // calculate child node bound
                fwd_bound += idx_bins[i].bound;
            for (int i = seg_bin_idx + 1; i < num_bins; i++)
                bwd_bound += idx_bins[i].bound;
        }
    } else { // equal primitive number split (two nodes have identical
             // primitives)
        std::vector<std::pair<float, int>> valued_indices;
        valued_indices.reserve(cur_node->size());
        for (int bvh_id : cur_node->prims) {
            valued_indices.emplace_back(bvh_infos[bvh_id].centroid[max_axis],
                                        bvh_id);
        }

        // Step 5: reordering the BVH info in the vector to make the segment
        // contiguous (keep around half of the bvh in lchild)
        int half_size = valued_indices.size() / 2;
        std::nth_element(
            valued_indices.begin(), valued_indices.begin() + half_size,
            valued_indices.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

        for (int i = 0; i < half_size; i++) {
            int bvh_id = valued_indices[i].second;
            lchild_idxs.push_back(bvh_id);
            fwd_bound += bvh_infos[bvh_id].bound;
        }
        for (int i = half_size; i < valued_indices.size(); i++) {
            int bvh_id = valued_indices[i].second;
            rchild_idxs.push_back(bvh_id);
            bwd_bound += bvh_infos[bvh_id].bound;
        }
        cur_node->bound += fwd_bound;
        cur_node->bound += bwd_bound;
        float split_cost =
            traverse_cost +
            (1.f / cur_node->bound.area()) *
                (fwd_bound.area() * float(half_size) +
                 bwd_bound.area() * float(valued_indices.size() - half_size));
        if (split_cost >= node_prim_cnt && prim_num < max_prim_node)
            lchild_idxs.clear();
    }

    if (!lchild_idxs.empty() &&
        !rchild_idxs.empty()) { // cost of splitting is less than making this
                                // node a leaf node
        cur_node->release();    // release mem for non-leaf nodes
        cur_node->lchild =
            new SBVHNode(std::move(fwd_bound), std::move(lchild_idxs));
        cur_node->rchild =
            new SBVHNode(std::move(bwd_bound), std::move(rchild_idxs));
        cur_node->axis = max_axis;

        int node_num = 1;
        node_num += recursive_sbvh_SAH(points1, points2, points3,
                                       flattened_idxs, cur_node->lchild,
                                       bvh_infos, depth + 1, max_prim_node);

        node_num += recursive_sbvh_SAH(points1, points2, points3,
                                       flattened_idxs, cur_node->rchild,
                                       bvh_infos, depth + 1, max_prim_node);
        return node_num;
    } else {
        return process_leaf();
    }
}

static int recursive_linearize(SBVHNode *cur_node, std::vector<float4> &nodes,
                               std::vector<CompactNode> &cache_nodes,
                               const int depth = 0,
                               const int cache_max_depth = 4) {
    // see the @ref bvh.cu for more information

    size_t current_size = nodes.size() >> 1,
           current_cached = cache_nodes.size();
    float4 node_f, node_b;
    cur_node->get_float4(node_f, node_b);
    nodes.push_back(node_f);
    nodes.push_back(node_b);
    reinterpret_cast<uint32_t &>(node_f.w) = 1;
    reinterpret_cast<uint32_t &>(node_b.w) = current_size;
    if (depth < cache_max_depth) {
        cache_nodes.emplace_back(node_f, node_b);
    }
    if (cur_node->non_leaf()) {
        // non-leaf node
        int lnodes = recursive_linearize(cur_node->lchild, nodes, cache_nodes,
                                         depth + 1, cache_max_depth);
        lnodes += recursive_linearize(cur_node->rchild, nodes, cache_nodes,
                                      depth + 1, cache_max_depth);
        INT_REF_CAST(nodes[2 * current_size + 1].w) = -(lnodes + 1);
        if (depth < cache_max_depth) {
            // store the jump offset to the next cached node (for non-leaf node)
            cache_nodes[current_cached].set_low_8bits(cache_nodes.size() -
                                                      current_cached);
        }
        return lnodes + 1; // include the cur_node
    } else {
        return 1;
    }
}

static SBVHNode *sbvh_root_start(const std::vector<Vec3> &points1,
                                 const std::vector<Vec3> &points2,
                                 const std::vector<Vec3> &points3,
                                 const Vec3 &world_min, const Vec3 &world_max,
                                 std::vector<int> &flattened_idxs,
                                 std::vector<BVHInfo> &bvh_infos, int &node_num,
                                 int max_prim_node = 16) {
    // Build BVH tree root node and start recursive tree construction
    printf("[SBVH] World min: ");
    print_vec3(world_min);
    printf("[SBVH] World max: ");
    print_vec3(world_max);
    SBVHNode *root_node = new SBVHNode(AABB(world_min, world_max, 0, 0), {});
    node_num = recursive_sbvh_SAH(points1, points2, points3, flattened_idxs,
                                  root_node, bvh_infos, max_prim_node);

    return root_node;
}

// Try to use two threads to build the BVH
void sbvh_build(const std::vector<Vec3> &points1,
                const std::vector<Vec3> &points2,
                const std::vector<Vec3> &points3,
                const std::vector<ObjInfo> &objects,
                const std::vector<int> &obj_med_idxs,
                const std::vector<bool> &sphere_flags, const Vec3 &world_min,
                const Vec3 &world_max, std::vector<int> &obj_idxs,
                std::vector<int> &prim_idxs, std::vector<float4> &nodes,
                std::vector<CompactNode> &cache_nodes, int &cache_max_level,
                const int max_prim_node, const float overlap_w) {
    bvh_overlap_w = overlap_w;
    std::vector<PrimMappingInfo> idx_prs;
    std::vector<BVHInfo> bvh_infos;
    int node_num = 0, num_prims_all = points1.size();
    index_input(objects, sphere_flags, idx_prs, num_prims_all);
    create_bvh_info(points1, points2, points3, idx_prs, obj_med_idxs,
                    bvh_infos);

    std::vector<int> flattened_idxs;
    // spatial split almost always ends up with more primitives
    flattened_idxs.reserve(num_prims_all * 2);
    SBVHNode *root_node =
        sbvh_root_start(points1, points2, points3, world_min, world_max,
                        flattened_idxs, bvh_infos, node_num, max_prim_node);

    printf("[SBVH] SBVH tree max depth: %d\n", max_depth);
    cache_max_level = std::min(std::max(max_depth - 1, 0), cache_max_level);
    nodes.reserve(node_num << 1);
    cache_nodes.reserve(1 << cache_max_level);

    // TODO(heqianyue): one last thing, re-organize prims, norms, uvs and object
    // maps

    printf("[SBVH] Number of nodes to cache: %llu (%d)\n", cache_nodes.size(),
           cache_max_level);

    // FIXME: MASK ALPHA, change obj_idxs
    obj_idxs.reserve(bvh_infos.size());
    prim_idxs.reserve(bvh_infos.size());
    for (BVHInfo &bvh : bvh_infos) {
        obj_idxs.emplace_back(bvh.bound.__bytes1);
        prim_idxs.emplace_back(bvh.bound.__bytes2);
    }
    delete root_node;
}
