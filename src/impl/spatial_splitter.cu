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
 * @brief SBVH spatial splitting utils
 * @date 2025.5.26
 */
#include "core/bvh_spatial.cuh"
#include <unordered_set>

static constexpr bool SSP_DEBUG = false;

template <int N>
void SpatialSplitter<N>::update_triangle(const Vec3 &v1, const Vec3 &v2,
                                         const Vec3 &v3, int prim_id) {
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

template <int N>
void SpatialSplitter<N>::update_bins(const std::vector<Vec3> &points1,
                                     const std::vector<Vec3> &points2,
                                     const std::vector<Vec3> &points3,
                                     /* possibly, add sphere flag later */
                                     const SBVHNode *const cur_node) {
    for (int prim_id : cur_node->prims) {
        update_triangle(points1[prim_id], points2[prim_id], points3[prim_id],
                        prim_id);
    }
}

template <int N>
float SpatialSplitter<N>::eval_spatial_split(const SBVHNode *const cur_node,
                                             int &seg_bin_idx,
                                             float traverse_cost) {
    float min_cost = 5e9f, node_prim_cnt = float(cur_node->prim_num());

    std::array<float, N> fwd_areas, bwd_areas;
    prim_cnts.fill(0);
    fwd_areas.fill(0);
    bwd_areas.fill(0);

    for (int i = 0; i < N; i++) {
        fwd_bound += bounds[i];
        prim_cnts[i] = enter_tris[i];
        if (i > 0)
            prim_cnts[i] += prim_cnts[i - 1];
        fwd_areas[i] = fwd_bound.area();
        if (i > 0) {
            bwd_bound += bounds[N - i];
            bwd_areas[N - 1 - i] = bwd_bound.area();
        }
    }
    float node_inv_area = 1. / cur_node->bound.area();

    for (int i = 0; i < N - 1; i++) {
        float cost = traverse_cost +
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
    const int prim_num = cur_node->prim_num();
    left_prims.reserve(prim_cnts[seg_bin_idx]);
    right_prims.reserve(prim_num / 2);
    std::unordered_set<int> exit_from_left;
    for (int i = 0; i <= seg_bin_idx; i++) {
        left_prims.insert(left_prims.begin(), enter_tris[i].begin(),
                          enter_tris[i].end());
        for (int v : exit_tris[i]) {
            exit_from_left.emplace(v);
        }
    }
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

    fwd_bound.clear();
    bwd_bound.clear();
    for (int i = 0; i <= seg_bin_idx; i++) // calculate child node bound
        fwd_bound += bounds[i];
    for (int i = seg_bin_idx + 1; i < N; i++)
        bwd_bound += bounds[i];
    return std::make_pair(fwd_bound, bwd_bound);
}
