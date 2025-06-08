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
#include <array>
#include <unordered_set>

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

    // for non-leaf node, release memory to reduce overhead
    void release() {
        prims.clear();
        prims.shrink_to_fit();
    }

    int base() const { return bound.base(); }
    int &base() { return bound.base(); }

    int prim_num() const { return bound.prim_cnt(); }
    int &prim_num() { return bound.prim_cnt(); }
    int size() const { return prims.size(); }

    bool non_leaf() const { return prims.empty(); }

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

class SBVHBuilderThreadSpan;

/**
 * @brief SBVH implementation
 */

template <int N> class SpatialSplitter {
  private:
    const bool employ_unsplit;
    const AABB bound;
    // split axis and range is not determined by SAH-BVH (not by centroids, but
    // by the extent of the AABB)
    const SplitAxis axis;
    const float s_pos;
    const float interval;

    std::array<AABB, N> bounds; // clipped binning results
    std::array<int, N> lprim_cnts;
    std::array<int, N> rprim_cnts;
    // this container holds the AABB of the bound-clipped triangles, introduced
    // to make reference unsplitting easier to compute
    std::vector<AABB> clip_poly_aabbs;
    std::unordered_set<int> unsplit_left;  // no adding to lchild set
    std::unordered_set<int> unsplit_right; // no adding to rchild set

  public:
    // ID of the triangles that enters the specified bin
    std::array<std::vector<int>, N> enter_tris;
    // ID of the triangles that exits the specified bin
    std::array<std::vector<int>, N> exit_tris;

  private:
    // given the vertices of a triangle, update the spatial splitter bins,
    // entering and exiting triangle records via one-sweep line-drawing-like
    // fast AABB bins update algorithm (chopped binning in the paper)
    bool update_triangle(std::vector<Vec3> &&points,
                         std::array<AABB, N> &bounds,
                         std::array<std::vector<int>, N> &enter_tris,
                         std::array<std::vector<int>, N> &exit_tris,
                         std::vector<AABB> &clip_poly_aabbs, int prim_id) const;

    void update_sphere(const Vec3 &center, float radius, int prim_id) {
        throw std::runtime_error(
            "SBVH with sphere primitive is not yet supported.");
    }

    inline int get_bin_id(const Vec3 &p) const {
        return std::clamp(static_cast<int>((p[axis] - s_pos) / interval), 0,
                          N - 1);
    }

    // after update_bins, the AABB of each bin might have parts outside of the
    // node AABB (for non split axis), we need to bound them
    void bound_all_bins();

  public:
    SpatialSplitter(const AABB &_bound, SplitAxis _axis, bool _unsplit = true)
        : employ_unsplit(_unsplit), bound(_bound), s_pos(_bound.mini[_axis]),
          interval((_bound.maxi[_axis] - _bound.mini[_axis]) /
                   static_cast<float>(N)),
          axis(_axis) {
        for (int i = 0; i < N; i++) {
            bounds[i].clear();
        }
        lprim_cnts.fill(0);
        rprim_cnts.fill(0);
    }

    bool employ_ref_unsplit() const noexcept { return this->employ_unsplit; }

    // given a node and the current BVHInfo vector, try to split the triangles
    // in the given range, the update is exact
    void update_bins(const std::vector<Vec3> &points1,
                     const std::vector<Vec3> &points2,
                     const std::vector<Vec3> &points3,
                     const SBVHBuilderThreadSpan &threads,
                     const SBVHNode *const cur_node);

    float eval_spatial_split(int &seg_bin_idx, int node_prim_cnt,
                             float traverse_cost);

    // optimize SAH cost by removing some of the duplicated references
    std::pair<AABB, AABB> apply_unsplit_reference(std::vector<int> &left_prims,
                                                  std::vector<int> &right_prims,
                                                  float &min_cost,
                                                  int seg_bin_idx);

    std::pair<AABB, AABB> apply_spatial_split(std::vector<int> &left_prims,
                                              std::vector<int> &right_prims,
                                              int seg_bin_idx);

    SplitAxis get_axis() const {
        return static_cast<SplitAxis>(this->axis | SplitAxis::SPATIAL_SPLIT);
    }
};

class SBVHBuilder {
  public:
    SBVHBuilder(std::array<std::vector<Vec3>, 3> &_vertices,
                std::array<std::vector<Vec3>, 3> &_normals,
                std::array<std::vector<Vec2>, 3> &_uvs,
                std::vector<bool> &_sphere_flags,
                std::vector<ObjInfo> &_objects, int _num_emitters,
                int _max_prim_node)
        : vertices(_vertices), normals(_normals), uvs(_uvs),
          sphere_flags(_sphere_flags), objects(_objects),
          num_emitters(_num_emitters), max_prim_node(_max_prim_node) {}

    void build(const std::vector<int> &obj_med_idxs, const Vec3 &world_min,
               const Vec3 &world_max, std::vector<int> &obj_idxs,
               std::vector<float4> &nodes,
               std::vector<CompactNode> &cache_nodes, int &cache_max_level,
               bool ref_unsplit = true);

    void post_process(std::vector<int> &obj_indices,
                      std::vector<int> &emitter_prims);

  private:
    std::array<std::vector<Vec3>, 3> &vertices;
    std::array<std::vector<Vec3>, 3> &normals;
    std::array<std::vector<Vec2>, 3> &uvs;
    std::vector<bool> &sphere_flags;
    std::vector<ObjInfo> &objects;
    const int num_emitters;
    const int max_prim_node;
    AABB root_bound;

    std::vector<int> flattened_idxs;
};
