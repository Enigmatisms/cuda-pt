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
 * @brief Spatial BVH utilities
 * @date 2025.5.25
 */

#pragma once
#include "core/bvh.cuh"
#include "core/constants.cuh"
#include "core/object.cuh"
#include <algorithm>
#include <array>

class SBVHNode {
  public:
    SBVHNode()
        : bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0), axis(AXIS_NONE),
          lchild(nullptr), rchild(nullptr) {
        prims.reserve(4);
    }

    SBVHNode(AABB &&_bound, std::vector<int> &&_prims)
        : lchild(nullptr), rchild(nullptr), bound(std::move(_bound)),
          axis(AXIS_NONE), prims(std::move(_prims)) {}

    ~SBVHNode() {
        if (lchild != nullptr)
            delete lchild;
        if (rchild != nullptr)
            delete rchild;
    }

    bool is_leaf() const { return lchild == nullptr; }

    int prim_num() const { return prims.size(); }

    // TODO: this might need to be refactored
    void get_float4(float4 &node_f, float4 &node_b) const {
        node_f.x = bound.mini.x();
        node_f.y = bound.mini.y();
        node_f.z = bound.mini.z();
        node_f.w = *reinterpret_cast<const float *>(&bound.__bytes1);
        node_b.x = bound.maxi.x();
        node_b.y = bound.maxi.y();
        node_b.z = bound.maxi.z();
        node_b.w = *reinterpret_cast<const float *>(&bound.__bytes2);
    }

  public:
    // The axis start and end are scaled up a little bit
    SplitAxis max_extent_axis(const std::vector<BVHInfo> &bvhs, float &min_r,
                              float &interval) const;

  public:
    AABB bound;
    SplitAxis axis;
    SBVHNode *lchild, *rchild;
    std::vector<int> prims; // SBVH will have duplicated indices
};

/**
 * @brief SBVH implementation
 */

template <int N> class SpatialSplitter {
  private:
    const SplitAxis axis;
    const float s_pos; // starting position
    const float e_pos; // ending position
    const float interval;

    std::vector<AABB> bounds;

    AABB fwd_bound, bwd_bound;
    std::array<int, N> prim_cnts; // cumsum of primitive cnts
  public:
    // ID of the triangles that enters the specified bin
    std::array<std::vector<int>, N> enter_tris;
    // ID of the triangles that exits the specified bin
    std::array<std::vector<int>, N> exit_tris;

  private:
    // given the vertices of a triangle, update the spatial splitter bins,
    // entering and exiting triangle records via one-sweep line-drawing-like
    // fast AABB bins update algorithm
    void update_triangle(const Vec3 &v1, const Vec3 &v2, const Vec3 &v3,
                         int prim_id);

    void update_sphere(const Vec3 &center, float radius, int prim_id) {
        throw std::runtime_error(
            "SBVH with sphere primitive is not yet supported.");
    }

    inline int get_bin_id(const Vec3 &p) const {
        return std::min(static_cast<int>(std::floor(
                            std::max(p[axis] - s_pos, 0) / interval)),
                        N - 1);
    }

  public:
    SpatialSplitter(float _s_pos, float _e_pos, SplitAxis _axis)
        : axis(_axis), s_pos(_s_pos), e_pos(_e_pos),
          interval((_e_pos - _s_pos) / static_cast<float>(N)) {}

    // given a node and the current BVHInfo vector, try to split the triangles
    // in the given range, the update is exact
    void update_bins(const std::vector<Vec3> &points1,
                     const std::vector<Vec3> &points2,
                     const std::vector<Vec3> &points3,
                     const SBVHNode *const cur_node);

    float eval_spatial_split(const SBVHNode *const cur_node, int &seg_bin_idx,
                             float traverse_cost);

    std::pair<AABB, AABB> apply_spatial_split(const SBVHNode *const cur_node,
                                              std::vector<int> &left_prims,
                                              std::vector<int> &right_prims,
                                              int seg_bin_idx);

    template <bool reverse>
    template <int N>
    AABB partial_sum(const int index) const {
        AABB result;
        result.clear();
        if constexpr (reverse) {
            for (int i = index + 1; i < N; i++) {
                result += bounds[i];
            }
        } else {
            for (int i = 0; i < index + 1; i++) {
                result += bounds[i];
            }
        }
    }

    AABB partial_sum(const int index, const bool reverse = false) const;
};

void sbvh_build(std::vector<Vec3> &points1, std::vector<Vec3> &points2,
                std::vector<Vec3> &points3, const std::vector<ObjInfo> &objects,
                const std::vector<int> &obj_med_idxs,
                std::vector<bool> &sphere_flags, const Vec3 &world_min,
                const Vec3 &world_max, std::vector<int> &obj_idxs,
                std::vector<int> &prim_idxs, std::vector<float4> &nodes,
                std::vector<CompactNode> &cache_nodes, int &max_cache_level,
                const int max_node_num, const float overlap_w) {}
