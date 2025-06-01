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
#include <unordered_set>

static constexpr int num_bins = 16;
static constexpr int num_sbins = 16; // spatial bins
static constexpr int no_div_threshold = 2;
static constexpr int sah_split_threshold = 8;
// A cluster with all the primitive centroid within a small range [less than
// 1e-3] is ill-posed. If there is more than 64 primitives, the primitives will
// be discarded
static constexpr bool SSP_DEBUG = false;
static constexpr float traverse_cost = 0.2f;
static constexpr float spatial_traverse_cost = 0.4f;
static constexpr int max_allowed_depth = 96;
static constexpr int same_size_split = 3;
static int max_depth = 0;

SplitAxis SBVHNode::max_extent_axis(const std::vector<BVHInfo> &bvhs,
                                    float &min_r, float &interval) const {

    Vec3 min_ctr = Vec3(std::numeric_limits<float>::max()),
         max_ctr = Vec3(-std::numeric_limits<float>::max());

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
void SpatialSplitter<N>::update_triangle(std::vector<Vec3> &&points,
                                         int prim_id) {
    // Note that spatial split triangles can have parts outside of the AABB
    // so we must not assume that AABB is tight (object-ly, but spatially)
    int min_axis_v = N, max_axis_v = -1;

    for (int i = 0; i < 3; i++) {
        Vec3 sp = points[i], ep = points[(i + 1) % 3];

        bool valid = bound.clip_line_segment(sp, ep, sp, ep);
        if (!valid)
            continue; // current edge out of range and will not affect bounding
                      // box
        if (sp[axis] > ep[axis])
            std::swap(sp, ep);

        Vec3 dir = ep - sp;
        float dim_v = dir[axis];
        int s_idx = get_bin_id(sp), e_idx = get_bin_id(ep);
        min_axis_v = std::min(min_axis_v, s_idx);
        max_axis_v = std::max(max_axis_v, e_idx);

        if (std::abs(dim_v) <= 1e-4f) { // parallel, extend directly
            bounds[s_idx].extend(sp);
            bounds[s_idx].extend(ep);
            continue;
        }
        dir *= 1.f / dim_v;

        float d2bin_start =
            s_pos + interval * static_cast<float>(s_idx) - sp[axis];
        Vec3 pt = sp + d2bin_start * dir;
        for (int id = s_idx; id <= e_idx; id++) {
            AABB &aabb = bounds[id];
            aabb.extend(id == s_idx ? sp : pt);
            pt += interval * dir;
            aabb.extend(id == e_idx ? ep : pt);
        }
    }
    enter_tris[min_axis_v].push_back(prim_id);
    exit_tris[max_axis_v].push_back(prim_id);
}

template <int N>
void SpatialSplitter<N>::update_bins(const std::vector<Vec3> &points1,
                                     const std::vector<Vec3> &points2,
                                     const std::vector<Vec3> &points3,
                                     /* possibly, add sphere flag later */
                                     const SBVHNode *const cur_node) {
    for (int prim_id : cur_node->prims) {
        update_triangle({points1[prim_id], points2[prim_id], points3[prim_id]},
                        prim_id);
    }
}

template <int N>
float SpatialSplitter<N>::eval_spatial_split(const SBVHNode *const cur_node,
                                             int &seg_bin_idx,
                                             float trav_cost) {
    float min_cost = 5e9f, node_prim_cnt = float(cur_node->size());

    std::array<float, N> fwd_areas, bwd_areas;
    prim_cnts.fill(0);
    fwd_areas.fill(0);
    bwd_areas.fill(0);

    AABB fwd_bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0),
        bwd_bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0);
    for (int i = 0; i < N; i++) {
        bounds[i].fix_degenerate(); // in case 0 width axis
        fwd_bound += bounds[i];
        prim_cnts[i] = enter_tris[i].size();
        fwd_areas[i] = fwd_bound.area();
        if (i > 0) {
            prim_cnts[i] += prim_cnts[i - 1];
            bwd_bound += bounds[N - i];
            bwd_areas[N - 1 - i] = bwd_bound.area();
        }
    }
    float node_inv_area = 1. / cur_node->bound.area();

    for (int i = 0; i < N - 1; i++) {
        float cost = trav_cost +
                     node_inv_area *
                         (float(prim_cnts[i]) * fwd_areas[i] +
                          (node_prim_cnt - float(prim_cnts[i])) * bwd_areas[i]);
        if (cost < min_cost) {
            min_cost = cost;
            seg_bin_idx = i;
        }
    }
    return min_cost;
}

template <int N>
std::pair<AABB, AABB> SpatialSplitter<N>::apply_spatial_split(
    const SBVHNode *const cur_node, std::vector<int> &left_prims,
    std::vector<int> &right_prims, int seg_bin_idx) {
    const int prim_num = cur_node->size();
    std::unordered_set<int> exit_from_left;
    left_prims.reserve(prim_cnts[seg_bin_idx]);
    exit_from_left.reserve(prim_num / 2);
    for (int i = 0; i <= seg_bin_idx; i++) {
        left_prims.insert(left_prims.begin(), enter_tris[i].begin(),
                          enter_tris[i].end());
        for (int v : exit_tris[i]) {
            exit_from_left.emplace(v);
        }
    }
    right_prims.reserve(prim_num - exit_from_left.size());
    for (int prim_id : cur_node->prims) {
        if (exit_from_left.count(prim_id))
            continue;
        right_prims.push_back(prim_id);
    }

    if constexpr (SSP_DEBUG) {
        if (left_prims.empty() || right_prims.empty()) {
            std::cerr << "Spatial split results in empty child nodes: "
                      << left_prims.size() << ", " << right_prims.size()
                      << std::endl;
            throw std::runtime_error("Spatial split failed.");
        }
    }

    AABB fwd_bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0),
        bwd_bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0);
    for (int i = 0; i <= seg_bin_idx; i++) // calculate child node bound
        fwd_bound += bounds[i];
    for (int i = seg_bin_idx + 1; i < N; i++)
        bwd_bound += bounds[i];
    return std::make_pair(fwd_bound, bwd_bound);
}

bool spatial_split_criteria(float root_area, float cur_area, float intrs_area,
                            int depth) {
    // SS can only be applied when depth >= the following
    static constexpr int spatial_split_depth = 0;
    // SS can be applied if local overlap >= the following
    static constexpr float local_overlap_factor = 0.5;
    // SS can be applied if overlap relative to root >= the following. This
    // factor is in fact mentioned in the original paper.
    static constexpr float root_overlap_factor = 1e-5f;

    return (depth > spatial_split_depth) &&
           ((intrs_area > cur_area * local_overlap_factor) ||
            (intrs_area > root_overlap_factor * root_area));
}

struct SplitInfo {
    int last_size;
    int counter;
    int depth;

    SplitInfo() : last_size(0), counter(0), depth(0) {}

    void update(int size) {
        if (size == last_size)
            return;
        last_size = size;
        counter = 0;
    }
};

// TODO(heqianyue): note that we currently don't support
// sphere primitive. Support it would be straightforward:
// overload the 'update' function for spheres
int recursive_sbvh_SAH(const std::vector<Vec3> &points1,
                       const std::vector<Vec3> &points2,
                       const std::vector<Vec3> &points3,
                       const std::vector<BVHInfo> &bvh_infos,
                       std::vector<int> &flattened_idxs,
                       SBVHNode *const cur_node, SplitInfo &&split_info,
                       float root_area, int max_prim_node = 16) {
    split_info.counter++;
    split_info.depth++;

    auto process_leaf = [&]() {
        // leaf node processing function
        cur_node->axis = AXIS_NONE;
        cur_node->base() = static_cast<int>(flattened_idxs.size());
        cur_node->prim_num() = static_cast<int>(cur_node->size());
        max_depth = std::max(max_depth, split_info.depth);
        for (int prim_id : cur_node->prims) {
            flattened_idxs.push_back(prim_id);
        }
        // TODO(heqianyue): check whether the leaf node has valid bound
        return 1;
    };

    if (cur_node->size() <= no_div_threshold ||
        split_info.counter >= same_size_split ||
        split_info.depth >= max_allowed_depth) {
        // if the node is small, or father nodes and child nodes share the same
        // size (spatial split duplication) for too many times, we'll create a
        // leaf node
        return process_leaf();
    }
    AABB fwd_bound(1e5f, -1e5f, 0, 0), bwd_bound(1e5f, -1e5f, 0, 0);
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

        float node_inv_area = 1. / fwd_bound.area();
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

        bool spatial_split_applied = false;
        if (spatial_split_criteria(root_area, cur_node->bound.area(),
                                   fwd_bound.intersection_area(bwd_bound),
                                   split_info.depth)) {
            // TODO(heqianyue): there are still some optimization that can be
            // implemented. (1) Reference unsplitting. Since split one primitive
            // reference into two nodes when the reference introduces little
            // overlap, we can unsplit the reference.

            SpatialSplitter<num_sbins> ssp(cur_node->bound);

            ssp.update_bins(points1, points2, points3, cur_node);

            int sbvh_seg_idx = 0;
            float sbvh_cost = ssp.eval_spatial_split(cur_node, sbvh_seg_idx,
                                                     spatial_traverse_cost);
            // printf("SBVH: spatial split cost: %f, object split cost: %f\n",
            // sbvh_cost, min_cost);
            if (sbvh_cost < min_cost &&
                (sbvh_cost < node_prim_cnt ||
                 prim_num > max_prim_node)) { // Spatial split should be applied
                min_cost = sbvh_cost;
                std::tie(fwd_bound, bwd_bound) = ssp.apply_spatial_split(
                    cur_node, lchild_idxs, rchild_idxs, sbvh_seg_idx);
                spatial_split_applied = true;
                max_axis = ssp.get_axis();
            }
        }

        if (!spatial_split_applied &&
            (min_cost < node_prim_cnt ||
             prim_num > max_prim_node)) { // object split
            fwd_bound.clear();
            bwd_bound.clear();
            for (int i = 0; i <= seg_bin_idx; i++) // calculate child node bound
                fwd_bound += idx_bins[i].bound;
            for (int i = seg_bin_idx + 1; i < num_bins; i++)
                bwd_bound += idx_bins[i].bound;
            // 1. SBVH is not applied ; 2. when the cost of splitting is lower
            // or 3. when there are more primitives than allowed We cannot
            // partition here, since partition will change the index of the BVH
            float pivot = min_range + interval * float(seg_bin_idx + 1);
            for (int bvh_id : cur_node->prims) {
                const BVHInfo &bvh = bvh_infos[bvh_id];
                if (bvh.centroid[max_axis] < pivot) {
                    lchild_idxs.push_back(bvh_id);
                } else {
                    rchild_idxs.push_back(bvh_id);
                }
            }
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
        if (split_cost >= node_prim_cnt && prim_num <= max_prim_node)
            lchild_idxs.clear();
    }

    if (!lchild_idxs.empty() &&
        !rchild_idxs.empty()) { // cost of splitting is less than making this
                                // node a leaf node
        cur_node->release();    // release mem for non-leaf nodes
        SplitInfo lchild_info = split_info, rchild_info = std::move(split_info);
        lchild_info.update(lchild_idxs.size());
        rchild_info.update(rchild_idxs.size());
        cur_node->lchild =
            new SBVHNode(std::move(fwd_bound), std::move(lchild_idxs));
        cur_node->rchild =
            new SBVHNode(std::move(bwd_bound), std::move(rchild_idxs));
        cur_node->axis = max_axis;

        int node_num = 1;
        node_num += recursive_sbvh_SAH(
            points1, points2, points3, bvh_infos, flattened_idxs,
            cur_node->lchild, std::move(lchild_info), root_area, max_prim_node);

        node_num += recursive_sbvh_SAH(
            points1, points2, points3, bvh_infos, flattened_idxs,
            cur_node->rchild, std::move(rchild_info), root_area, max_prim_node);
        return node_num;
    } else {
        return process_leaf();
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
    std::vector<int> all_prims(points1.size());
    std::iota(all_prims.begin(), all_prims.end(), 0);
    SBVHNode *root_node = new SBVHNode(
        AABB(world_min, world_max, 0, points1.size()), std::move(all_prims));
    node_num = recursive_sbvh_SAH(points1, points2, points3, bvh_infos,
                                  flattened_idxs, root_node, SplitInfo(),
                                  root_node->bound.area(), max_prim_node);

    return root_node;
}

template <typename ContainerTy, size_t Dim = 3>
void remap_helper_func(const std::vector<int> &flattened_idxs,
                       ContainerTy &source) {
    static constexpr int n_threads = 4;
    const size_t num_new_prims = flattened_idxs.size();
    const size_t padded_size =
        (num_new_prims + n_threads - 1) / n_threads; // workload for each thread

    ContainerTy mapped_vals;
    if constexpr (Dim == 1) {
        mapped_vals.resize(num_new_prims);
    } else {
        for (int i = 0; i < 3; i++) {
            mapped_vals[i].resize(num_new_prims);
        }
    }
#pragma omp parallel for num_threads(n_threads)
    for (int tid = 0; tid < n_threads; tid++) {
        const size_t s_pos = tid * padded_size,
                     e_pos = std::min(s_pos + padded_size, num_new_prims);
        if constexpr (Dim == 1) {
            for (size_t i = s_pos; i < e_pos; i++) {
                int index = flattened_idxs[i];
                mapped_vals[i] = source[index];
            }
        } else {
#pragma unroll
            for (int dim = 0; dim < Dim; dim++) {
                const auto &old_vec = source[dim];
                auto &new_vec = mapped_vals[dim];
                for (size_t i = s_pos; i < e_pos; i++) {
                    int index = flattened_idxs[i];
                    new_vec[i] = old_vec[index];
                }
            }
        }
    }
    source = std::move(mapped_vals);
}

void SBVHBuilder::post_process(std::vector<int> &obj_indices,
                               std::vector<int> &emitter_prims) {
    // remap all the vertices, normals, UVs and object indices for SBVH. There
    // are two major step for this: (1) reordered vertices, normals, UVs, object
    // index and sphere_flags using an multi-threading approach (or SIMD). (2)
    // Deal with the emissive primitives (remove duplication)
    size_t original_size = vertices[0].size();
    remap_helper_func(flattened_idxs, vertices);
    remap_helper_func(flattened_idxs, normals);
    remap_helper_func(flattened_idxs, uvs);
    remap_helper_func<std::vector<int>, 1>(flattened_idxs, obj_indices);
    remap_helper_func<std::vector<bool>, 1>(flattened_idxs, sphere_flags);

    const size_t num_prims = flattened_idxs.size();
    std::vector<std::vector<int>> eprim_idxs(num_emitters);
    std::vector<bool> visited(original_size, false);
    for (int i = 0; i < num_prims; i++) {
        // skip duplicated emissive primitives, if the duplicated primitives are
        // not skipped over, the emissive primitive sampling will be biased so
        // the emissive indices should be unique
        int origin_prim_id = flattened_idxs[i];
        if (visited[origin_prim_id])
            continue;
        visited[origin_prim_id] = true;

        int obj_idx = obj_indices[i] & 0x000fffff;
        const auto &object = objects[obj_idx];
        if (object.is_emitter()) {
            int emitter_idx = object.emitter_id - 1;
            eprim_idxs[emitter_idx].push_back(i);
        }
    }

    std::vector<int> e_prim_offsets;
    e_prim_offsets.push_back(0);
    for (const auto &eprim_idx : eprim_idxs) {
        e_prim_offsets.push_back(eprim_idx.size());
        for (int index : eprim_idx)
            emitter_prims.push_back(index);
    }
    std::partial_sum(e_prim_offsets.begin(), e_prim_offsets.end(),
                     e_prim_offsets.begin());
    for (ObjInfo &obj : objects) {
        if (!obj.is_emitter())
            continue;
        obj.prim_offset = e_prim_offsets[obj.emitter_id - 1];
    }
}

// Try to use two threads to build the BVH
void SBVHBuilder::build(const std::vector<int> &obj_med_idxs,
                        const Vec3 &world_min, const Vec3 &world_max,
                        std::vector<int> &obj_idxs, std::vector<float4> &nodes,
                        std::vector<CompactNode> &cache_nodes,
                        int &cache_max_level) {
    const auto &points1 = vertices[0], &points2 = vertices[1],
               &points3 = vertices[2];

    std::vector<PrimMappingInfo> idx_prs;
    std::vector<BVHInfo> bvh_infos;
    int node_num = 0, num_prims_all = points1.size();
    BVHBuilder::index_input(objects, sphere_flags, idx_prs, num_prims_all);
    BVHBuilder::create_bvh_info(points1, points2, points3, idx_prs,
                                obj_med_idxs, bvh_infos);

    // spatial split almost always ends up with more primitives
    flattened_idxs.reserve(num_prims_all * 2);
    SBVHNode *root_node =
        sbvh_root_start(points1, points2, points3, world_min, world_max,
                        flattened_idxs, bvh_infos, node_num, max_prim_node);

    printf("[SBVH] SBVH tree max depth: %d, duplicated primitives: %d (%d)\n",
           max_depth, flattened_idxs.size(), points1.size());
    float total_cost =
        calculate_cost(root_node, traverse_cost, spatial_traverse_cost);
    printf("[SBVH] Traversed BVH SAH cost: %.7f, AVG: %.7f\n", total_cost,
           total_cost / static_cast<float>(bvh_infos.size()));
    cache_max_level = std::min(std::max(max_depth - 1, 0), cache_max_level);
    nodes.reserve(node_num << 1);
    cache_nodes.reserve(1 << cache_max_level);

    recursive_linearize(root_node, nodes, cache_nodes, 0);
    printf("[SBVH] Number of nodes to cache: %lu (%d)\n", cache_nodes.size(),
           cache_max_level);

    obj_idxs.reserve(bvh_infos.size());
    for (BVHInfo &bvh : bvh_infos) {
        obj_idxs.emplace_back(bvh.bound.__bytes1);
    }
    delete root_node;
}
