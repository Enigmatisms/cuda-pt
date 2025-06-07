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
#include "core/proc_geometry.cuh"
#include "core/stats.h"
#include <algorithm>
#include <cassert>
#include <numeric>

static constexpr int num_bins = 64;
static constexpr int num_sbins = 128; // spatial bins
static constexpr int no_div_threshold = 2;
static constexpr int sah_split_threshold = 8;
// A cluster with all the primitive centroid within a small range [less than
// 1e-3] is ill-posed. If there is more than 64 primitives, the primitives will
// be discarded
static constexpr bool SSP_DEBUG = true;
static constexpr float traverse_cost = 0.2f;
static constexpr float spatial_traverse_cost = 0.21f;
static constexpr int max_allowed_depth = 96;
// when number of triangles to process is greater than the following,
// `update_bin` will employ thread pool to accelerate binning
static constexpr int workload_threshold = 128;
#ifdef OPENMP_ENABLED
#define OMP_GET_THREAD_ID omp_get_thread_num()
static constexpr int number_of_workers = 8;
#else
#define OMP_GET_THREAD_ID 0
static constexpr int number_of_workers = 1;
#endif // OPENMP_ENABLED
static int max_depth = 0;

SplitAxis SBVHNode::max_extent_axis(const std::vector<BVHInfo> &bvhs,
                                    float &min_r, float &interval) const {
    Vec3 min_ctr = Vec3(AABB_INVALID_DIST), max_ctr = Vec3(-AABB_INVALID_DIST);

    // Note: SBVH requires that the AABB of the child node to be <= to the
    // father's AABB. Therefore, if there is a spatial split followed by an
    // object split, some parts of the primitives might be outside of the AABB
    // of the father node. As a result, the centroids, and even the bins might
    // be outside of the AABB of the father node. So, for max extent and object
    // spatial binning, we need to clip the range inside the AABB of the father
    // node . This is different from BVH nodes (since the AABBs of their child
    // nodes must reside within the AABBs of the father nodes)
    for (int bvh_id : prims) {
        Vec3 ctr = bound.clamp(bvhs[bvh_id].centroid);
        min_ctr.minimized(ctr);
        max_ctr.maximized(ctr);
    }

    Vec3 diff = max_ctr - min_ctr;
    float max_diff = diff.x();
    min_r = min_ctr[0] - AABB_EPS;
    int split_axis = 0;
    if (diff.y() > max_diff) {
        max_diff = diff.y();
        split_axis = 1;
        min_r = min_ctr[1] - AABB_EPS;
    }
    if (diff.z() > max_diff) {
        max_diff = diff.z();
        split_axis = 2;
        min_r = min_ctr[2] - AABB_EPS;
    }
    if (diff.max_elem() < 1e-3) {
        return SplitAxis::AXIS_NONE;
    }
    interval = (max_diff + AABB_EPS * 2.f) / float(num_bins);
    return SplitAxis(split_axis);
}

template <int N> void SpatialSplitter<N>::bound_all_bins() {
    for (int i = 0; i < N; i++) {
        bounds[i] ^= bound;
    }
}

template <int N>
bool SpatialSplitter<N>::update_triangle(
    std::vector<Vec3> &&points, std::array<AABB, N> &_bounds,
    std::array<std::vector<int>, N> &_enter_tris,
    std::array<std::vector<int>, N> &_exit_tris,
    std::vector<AABB> &_clip_poly_aabbs, int prim_id) const {
    // Note that spatial split triangles can have parts outside of the AABB
    // so we must not assume that AABB is tight (object-ly, but spatially)

    int min_axis_v = N, max_axis_v = -1;

    std::vector<Vec3> clipped_poly =
        aabb_triangle_clipping(bound, std::move(points));

    if (clipped_poly.empty()) {
        if constexpr (SSP_DEBUG) {
            std::cerr << "[SBVH Warn] Primitive " << prim_id
                      << " discarded due to being degenerated after triangle "
                         "clipping. This should not happen.\n";
        }
        return false;
    }

    AABB clip_aabb(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0);
    Vec3 sp = clipped_poly.back();
    for (size_t i = 0; i < clipped_poly.size(); i++) {
        Vec3 ep = clipped_poly[i], old_ep = ep;
        if (employ_unsplit)
            clip_aabb.extend(ep);

        if (sp[axis] > ep[axis])
            std::swap(sp, ep);

        Vec3 dir = ep - sp;
        float dim_v = dir[axis];
        int s_idx = get_bin_id(sp), e_idx = get_bin_id(ep);
        min_axis_v = std::min(min_axis_v, s_idx);
        max_axis_v = std::max(max_axis_v, e_idx);

        if (std::abs(dim_v) < 1e-5f) {
            _bounds[s_idx].extend(sp);
            _bounds[e_idx].extend(ep);
        } else {
            dir *= 1.f / dim_v;
            float d2bin_start =
                s_pos + interval * static_cast<float>(s_idx) - sp[axis];
            Vec3 pt = sp.advance(dir, d2bin_start);
            for (int id = s_idx; id <= e_idx; id++) {
                AABB &aabb = _bounds[id];
                aabb.extend(bound.clamp(s_idx == id ? sp : pt));
                pt = pt.advance(dir, interval);
                aabb.extend(bound.clamp(e_idx == id ? ep : pt));
            }
        }
        sp = std::move(old_ep);
    }

    if (employ_unsplit) {
        auto packed_indices = reinterpret_cast<int16_t *>(&clip_aabb.__bytes1);
        packed_indices[0] = min_axis_v;
        packed_indices[1] = max_axis_v;
        clip_aabb.__bytes2 = prim_id;
        _clip_poly_aabbs.emplace_back(std::move(clip_aabb));
    }
    _enter_tris[min_axis_v].push_back(prim_id);
    _exit_tris[max_axis_v].push_back(prim_id);
    return true;
}

// declared for parallel processing
struct ChoppedBinningData {
    std::array<AABB, num_sbins> bounds;
    std::array<std::vector<int>, num_sbins> enter_tris;
    std::array<std::vector<int>, num_sbins> exit_tris;
    std::vector<AABB> clip_poly_aabbs;

    ChoppedBinningData() {
        for (int i = 0; i < num_sbins; i++) {
            bounds[i].clear();
        }
    }
};

template <int N>
void SpatialSplitter<N>::update_bins(const std::vector<Vec3> &points1,
                                     const std::vector<Vec3> &points2,
                                     const std::vector<Vec3> &points3,
                                     /* possibly, add sphere flag later */
                                     const SBVHNode *const cur_node) {
    // the following can be made faster by partitioning and multi-threading
    if (cur_node->size() > workload_threshold) {
        // multi-thread implementation
        std::array<ChoppedBinningData, number_of_workers> all_data;

#pragma omp parallel for num_threads(number_of_workers)
        for (size_t i = 0; i < cur_node->size(); i++) {
            int prim_id = cur_node->prims[i];
            int thread_id = OMP_GET_THREAD_ID;
            ChoppedBinningData &local_data = all_data[thread_id];
            update_triangle(
                {points1[prim_id], points2[prim_id], points3[prim_id]},
                local_data.bounds, local_data.enter_tris, local_data.exit_tris,
                local_data.clip_poly_aabbs, prim_id);
        }

        ChoppedBinningData result;
        size_t clip_aabb_size = 0;
        for (ChoppedBinningData &local_data : all_data) {
            for (int bin_id = 0; bin_id < N; bin_id++) {
                result.bounds[bin_id] += local_data.bounds[bin_id];
                result.enter_tris[bin_id].insert(
                    result.enter_tris[bin_id].end(),
                    local_data.enter_tris[bin_id].begin(),
                    local_data.enter_tris[bin_id].end());
                result.exit_tris[bin_id].insert(
                    result.exit_tris[bin_id].end(),
                    local_data.exit_tris[bin_id].begin(),
                    local_data.exit_tris[bin_id].end());
            }
            if (employ_unsplit) {
                clip_aabb_size += local_data.clip_poly_aabbs.size();
            }
        }
        if (clip_aabb_size > 0) {
            result.clip_poly_aabbs.reserve(clip_aabb_size);
            for (ChoppedBinningData &local_data : all_data) {
                result.clip_poly_aabbs.insert(
                    result.clip_poly_aabbs.end(),
                    local_data.clip_poly_aabbs.begin(),
                    local_data.clip_poly_aabbs.end());
            }
        }
        bounds = std::move(result.bounds);
        enter_tris = std::move(result.enter_tris);
        exit_tris = std::move(result.exit_tris);
        clip_poly_aabbs = std::move(result.clip_poly_aabbs);
    } else {
        // single-threaded implement
        for (int prim_id : cur_node->prims) {
            update_triangle(
                {points1[prim_id], points2[prim_id], points3[prim_id]}, bounds,
                enter_tris, exit_tris, clip_poly_aabbs, prim_id);
        }
    }
}

template <int N>
float SpatialSplitter<N>::eval_spatial_split(int &seg_bin_idx,
                                             int node_prim_cnt,
                                             float trav_cost) {
    float min_cost = 5e9f;

    std::array<float, N> fwd_areas, bwd_areas;
    fwd_areas.fill(0);
    bwd_areas.fill(0);

    AABB fwd_bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0),
        bwd_bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0);
    for (int i = 0; i < N; i++) {
        fwd_bound += bounds[i];
        fwd_areas[i] = fwd_bound.area();
        lprim_cnts[i] = enter_tris[i].size();
        if (i > 0) {
            lprim_cnts[i] += lprim_cnts[i - 1];
            bwd_bound += bounds[N - i];
            bwd_areas[N - 1 - i] = bwd_bound.area();
            // the same as BVH, the [N-1] will be 0, since seg_idx can never be
            // N - 1, also, exit_tris[0] will not be accessed (since
            // unnecessary)
            rprim_cnts[N - 1 - i] = exit_tris[N - i].size() + rprim_cnts[N - i];
        }
    }
    float node_inv_area = 1.f / bound.area();

    for (int i = 0; i < N - 1; i++) {
        float cost =
            trav_cost + node_inv_area * (float(lprim_cnts[i]) * fwd_areas[i] +
                                         float(rprim_cnts[i]) * bwd_areas[i]);
        if (cost < min_cost) {
            min_cost = cost;
            seg_bin_idx = i;
        }
    }
    return min_cost;
}

template <int N>
std::pair<AABB, AABB>
SpatialSplitter<N>::apply_unsplit_reference(std::vector<int> &left_prims,
                                            std::vector<int> &right_prims,
                                            float &min_cost, int seg_bin_idx) {
    // the min_cost is not a standard SAH cost. min_cost(here) = (min_cost -
    // traverse_cost) / node_inv_area;
    int lchild_cnt = lprim_cnts[seg_bin_idx],
        rchild_cnt = rprim_cnts[seg_bin_idx];
    // for child node with only one primitive, no need for unsplitting

    AABB fwd_bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0),
        bwd_bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0);
    for (int i = 0; i <= seg_bin_idx; i++) // calculate child node bound
        fwd_bound += bounds[i];
    for (int i = seg_bin_idx + 1; i < N; i++)
        bwd_bound += bounds[i];
    for (size_t i = 0; i < clip_poly_aabbs.size(); i++) {
        if (lchild_cnt <= 1 || rchild_cnt <= 1)
            break;
        const AABB &aabb = clip_poly_aabbs[i];
        // if not a straddled reference, skip
        auto packed_indices = reinterpret_cast<const int16_t *>(&aabb.__bytes1);
        if (packed_indices[0] > seg_bin_idx || packed_indices[1] <= seg_bin_idx)
            continue;
        float to_left_cost = (fwd_bound + aabb).area() * lchild_cnt +
                             bwd_bound.area() * (rchild_cnt - 1),
              to_right_cost = fwd_bound.area() * (lchild_cnt - 1) +
                              (aabb + bwd_bound).area() * rchild_cnt;
        if (to_left_cost >= min_cost && to_right_cost >= min_cost)
            continue;
        int index = aabb.__bytes2;
        if (to_left_cost < to_right_cost) { // the less one must < min_cost
            fwd_bound += aabb;
            min_cost = to_left_cost;
            unsplit_right.emplace(index);
            rchild_cnt--;
        } else {
            bwd_bound += aabb;
            min_cost = to_right_cost;
            unsplit_left.emplace(index);
            lchild_cnt--;
        }
    }
#define FILTER_EMPLACE(src, dst, filter, cnt, begin_i, end_i)                  \
    dst.reserve(cnt);                                                          \
    for (int i = begin_i; i <= end_i; i++) {                                   \
        const auto &idxs = src[i];                                             \
        for (int prim_idx : idxs) {                                            \
            if (filter.count(prim_idx))                                        \
                continue;                                                      \
            dst.push_back(prim_idx);                                           \
        }                                                                      \
    }

    FILTER_EMPLACE(enter_tris, left_prims, unsplit_left, lchild_cnt, 0,
                   seg_bin_idx)
    FILTER_EMPLACE(exit_tris, right_prims, unsplit_right, rchild_cnt,
                   seg_bin_idx + 1, N - 1)
#undef FILTER_EMPLACE
    return std::make_pair(fwd_bound, bwd_bound);
}

template <int N>
std::pair<AABB, AABB>
SpatialSplitter<N>::apply_spatial_split(std::vector<int> &left_prims,
                                        std::vector<int> &right_prims,
                                        int seg_bin_idx) {
    left_prims.reserve(lprim_cnts[seg_bin_idx]);
    for (int i = 0; i <= seg_bin_idx; i++) {
        left_prims.insert(left_prims.end(), enter_tris[i].begin(),
                          enter_tris[i].end());
    }
    right_prims.reserve(rprim_cnts[seg_bin_idx]);
    for (int i = seg_bin_idx + 1; i < N; i++) {
        right_prims.insert(right_prims.end(), exit_tris[i].begin(),
                           exit_tris[i].end());
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

bool spatial_split_criteria(float root_area, float intrs_area, int num_prims) {
    // SS can be applied if overlap relative to root >= the following. This
    // factor is in fact mentioned in the original paper.
    static constexpr float root_overlap_factor = 1e-5f;

    return intrs_area > root_overlap_factor * root_area;
}

// TODO(heqianyue): note that we currently don't support
// sphere primitive. Support it would be straightforward:
// overload the 'update' function for spheres
int recursive_sbvh_SAH(const std::vector<Vec3> &points1,
                       const std::vector<Vec3> &points2,
                       const std::vector<Vec3> &points3,
                       const std::vector<BVHInfo> &bvh_infos,
                       std::vector<int> &flattened_idxs,
                       SBVHNode *const cur_node, int depth, float root_area,
                       int max_prim_node = 16, bool ref_unsplit = true) {
    auto process_leaf = [&]() {
        // leaf node processing function
        cur_node->axis = AXIS_NONE;
        cur_node->base() = static_cast<int>(flattened_idxs.size());
        cur_node->prim_num() = static_cast<int>(cur_node->size());
        max_depth = std::max(max_depth, depth);
        for (int prim_id : cur_node->prims) {
            flattened_idxs.push_back(prim_id);
        }
        return 1;
    };
    float min_range = 0, interval = 0;
    // Step 1: decide the axis that expands the maximum extent of space
    SplitAxis max_axis =
        cur_node->max_extent_axis(bvh_infos, min_range, interval);

    if (cur_node->size() <= no_div_threshold || depth >= max_allowed_depth ||
        max_axis == SplitAxis::AXIS_NONE) {
        // if the node is small, or father nodes and child nodes share the same
        // size (spatial split duplication) for too many times, we'll create a
        // leaf node
        return process_leaf();
    }
    AABB fwd_bound(1e5f, -1e5f, 0, 0), bwd_bound(1e5f, -1e5f, 0, 0);
    const int prim_num = cur_node->size();
    float min_cost = 5e9f, node_prim_cnt = float(prim_num);

    std::vector<int> lchild_idxs, rchild_idxs;
    if (prim_num > sah_split_threshold) { // SAH
        // Step 2: binning the space
        std::array<AxisBins, num_bins> idx_bins;
        for (int bvh_id : cur_node->prims) {
            // some of the primitives might just have their centroids outside of
            // all the bins (as a result from spatial split followed by an
            // object split)
            int index = std::clamp(
                static_cast<int>(
                    (bvh_infos[bvh_id].centroid[max_axis] - min_range) /
                    interval),
                0, num_bins - 1);
            idx_bins[index].push(bvh_infos[bvh_id]);
        }
        for (int i = 0; i < num_bins; i++) {
            idx_bins[i].bound ^= cur_node->bound;
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

        fwd_bound.clear();
        bwd_bound.clear();
        for (int i = 0; i <= seg_bin_idx; i++) // calculate child node bound
            fwd_bound += idx_bins[i].bound;
        for (int i = seg_bin_idx + 1; i < num_bins; i++)
            bwd_bound += idx_bins[i].bound;

        bool spatial_split_applied = false;
        if (spatial_split_criteria(
                root_area, fwd_bound.intersection_area(bwd_bound), prim_num)) {
            SpatialSplitter<num_sbins> ssp(cur_node->bound, max_axis,
                                           ref_unsplit);
            ssp.update_bins(points1, points2, points3, cur_node);

            int sbvh_seg_idx = 0;
            float sbvh_cost = ssp.eval_spatial_split(
                sbvh_seg_idx, cur_node->size(), spatial_traverse_cost);
            // printf("SBVH: spatial split cost: %f, object split cost: %f\n",
            // sbvh_cost, min_cost);
            if (sbvh_cost < min_cost &&
                sbvh_cost < node_prim_cnt) { // Spatial split, actually node num
                                             // is not capped here
                max_axis = ssp.get_axis();
                if (ssp.employ_ref_unsplit()) {
                    sbvh_cost = (sbvh_cost - spatial_traverse_cost) *
                                cur_node->bound.area();
                    float old_sbvh_cost = sbvh_cost;
                    std::tie(fwd_bound, bwd_bound) =
                        ssp.apply_unsplit_reference(lchild_idxs, rchild_idxs,
                                                    sbvh_cost, sbvh_seg_idx);
                    if (old_sbvh_cost > sbvh_cost + THP_EPS) {
                        reinterpret_cast<int &>(max_axis) |=
                            SplitAxis::REF_UNSPLIT;
                    }
                } else {
                    std::tie(fwd_bound, bwd_bound) = ssp.apply_spatial_split(
                        lchild_idxs, rchild_idxs, sbvh_seg_idx);
                }
                fwd_bound.grow(1e-5f);
                bwd_bound.grow(1e-5f);
                spatial_split_applied = true;
            }
        }

        if (!spatial_split_applied &&
            (min_cost < node_prim_cnt ||
             prim_num > max_prim_node)) { // object split
            // 1. SBVH is not applied ; 2. when the cost of splitting is lower
            // or 3. when there are more primitives than allowed We cannot
            // partition here, since partition will change the index of the BVH
            lchild_idxs.reserve(prim_num / 2);
            rchild_idxs.reserve(prim_num / 2);
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
        fwd_bound ^= cur_node->bound;
        bwd_bound ^= cur_node->bound;
        float split_cost =
            traverse_cost +
            (1.f / cur_node->bound.area()) *
                (fwd_bound.area() * float(half_size) +
                 bwd_bound.area() * float(valued_indices.size() - half_size));
        if (split_cost >= node_prim_cnt && prim_num <= max_prim_node)
            fwd_bound.clear();
    }

    if (!lchild_idxs.empty() && !rchild_idxs.empty() && fwd_bound.is_valid() &&
        bwd_bound.is_valid()) {
        // in no case should the child node bound exceeds the father bound
        cur_node->release(); // release mem for non-leaf nodes
        cur_node->lchild =
            new SBVHNode(std::move(fwd_bound), std::move(lchild_idxs));
        cur_node->rchild =
            new SBVHNode(std::move(bwd_bound), std::move(rchild_idxs));
        cur_node->axis = max_axis;

        int node_num = 1;
        node_num += recursive_sbvh_SAH(
            points1, points2, points3, bvh_infos, flattened_idxs,
            cur_node->lchild, depth + 1, root_area, max_prim_node, ref_unsplit);

        node_num += recursive_sbvh_SAH(
            points1, points2, points3, bvh_infos, flattened_idxs,
            cur_node->rchild, depth + 1, root_area, max_prim_node, ref_unsplit);
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
                                 int max_prim_node = 16,
                                 bool ref_unsplit = true) {
    // Build BVH tree root node and start recursive tree construction
    printf("[SBVH] World min: ");
    print_vec3(world_min);
    printf("[SBVH] World max: ");
    print_vec3(world_max);
    std::vector<int> all_prims(points1.size());
    std::iota(all_prims.begin(), all_prims.end(), 0);
    SBVHNode *root_node = new SBVHNode(
        AABB(world_min, world_max, 0, points1.size()), std::move(all_prims));
    node_num = recursive_sbvh_SAH(
        points1, points2, points3, bvh_infos, flattened_idxs, root_node, 0,
        root_node->bound.area(), max_prim_node, ref_unsplit);

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
                        int &cache_max_level, bool ref_unsplit) {
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
    SBVHNode *root_node = sbvh_root_start(points1, points2, points3, world_min,
                                          world_max, flattened_idxs, bvh_infos,
                                          node_num, max_prim_node, ref_unsplit);

    printf("[SBVH] SBVH tree max depth: %d, duplicated primitives: %lu (%lu)\n",
           max_depth, flattened_idxs.size(), points1.size());
    float total_cost =
        calculate_cost(root_node, traverse_cost, spatial_traverse_cost);
    printf("[SBVH] Traversed BVH SAH cost: %.7f, AVG: %.7f\n", total_cost,
           total_cost / static_cast<float>(bvh_infos.size()));
    calculate_tree_metrics(root_node);

    cache_max_level = std::min(std::max(max_depth - 1, 0), cache_max_level);
    nodes.reserve(node_num << 1);
    cache_nodes.reserve(1 << cache_max_level);

    recursive_linearize(root_node, nodes, cache_nodes, 0, cache_max_level);

    printf("[SBVH] Number of nodes to cache: %lu (%d)\n", cache_nodes.size(),
           cache_max_level);
    // only for debug: level_order_traverse(root_node, 8);

    obj_idxs.reserve(bvh_infos.size());
    for (BVHInfo &bvh : bvh_infos) {
        obj_idxs.emplace_back(bvh.bound.__bytes1);
    }
    delete root_node;
}
