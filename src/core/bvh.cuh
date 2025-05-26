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
 * @brief BVH utilities
 * @date Unknown
 */

#pragma once
#include "core/constants.cuh"
#include "core/object.cuh"
#include <algorithm>

enum SplitAxis : int { AXIS_X, AXIS_Y, AXIS_Z, AXIS_NONE };

struct BVHInfo {
    // BVH is for both triangle meshes and spheres
    AABB bound;
    Vec3 centroid;

    BVHInfo() : bound(1e5f, -1e5f, 0, 0), centroid() {}
    BVHInfo(const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, int obj_idx,
            int prim_idx, bool is_sphere = false) {
        // Extract two vertices for the primitive, once converted to AABB
        // We don't need to distinguish between mesh or sphere
        // Note that vertex vectors in the primitive matrix are col-major order
        if (is_sphere) {
            centroid = p1;
            bound = AABB(p1 - p2.x() - AABB_EPS, p1 + p2.x() + AABB_EPS,
                         obj_idx, prim_idx);
        } else {
            centroid = (p1 + p2 + p3) * 0.33333333333f; // barycenter
            bound = AABB(p1, p2, p3, obj_idx, prim_idx);
        }
    }

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
};

class BVHNode {
  public:
    BVHNode()
        : bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0), axis(AXIS_NONE),
          lchild(nullptr), rchild(nullptr) {}
    BVHNode(int base, int prim_num)
        : bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, base, prim_num),
          axis(AXIS_NONE), lchild(nullptr), rchild(nullptr) {}
    ~BVHNode() {
        if (lchild != nullptr)
            delete lchild;
        if (rchild != nullptr)
            delete rchild;
    }

    bool is_leaf() const { return lchild == nullptr; }

    int base() const { return bound.base(); }
    int &base() { return bound.base(); }

    int prim_num() const { return bound.prim_cnt(); }
    int &prim_num() { return bound.prim_cnt(); }

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
    BVHNode *lchild, *rchild;
};

/**
 * @note @attention
 * For taichi lang, there is actually a huge problem:
 * To traverse the BVH tree, it is best to choose between lchild and rchild
 * according to split axis and the current ray direction, since for example
 * if ray points along neg-x-axis, and the node is split along x axis
 * lchild contains smaller x-coordinate nodes and the otherwise for rchild
 * we should of course first traverse the rchild. BUT, taichi lang has neither
 * (1) kernel stack nor (2) thread local dynamic memory allocation, therefore
 * neither can we record whether the node is accessed nor store the node to be
 * accessed in a stack. Allocate a global field which might incur (H * W * D *
 * 4) B memory consumption, where D is the depth of BVH tree, which can be
 * extremely imbalanced when the primitives are distributed with poor
 * uniformity. Therefore, in this implementation, we can only opt for a
 * suboptimal solution, to traverse the tree using just DFS, resulting in a
 * simple traversal method: the index for the linear node container is
 * monotonously increasing. So in this implementation, split axis will not be
 * included (which will not be used anyway).
 * TODO: we can opt for dynamic snode, but I think this would be ugly. Simple
 * unidirectional traversal would be much better than brute-force traversal with
 * only AABB pruning.
 *
 * This is actually borrowed from my AdaPT repo. Now, for CUDA, things are the
 * same: dynamically indexing a local array will have the data winding up in the
 * GMEM, which can be slow So, I will still implement a jump-back free version,
 * with no stack
 */
class LinearNode {
  public:
    CPT_CPU_GPU LinearNode() : aabb(1e5f, -1e5f, 0, 0) {}

    // linear nodes are initialized during DFS binary tree traversal
    CPT_CPU LinearNode(const BVHNode *const bvh)
        : aabb(bvh->bound.mini, bvh->bound.maxi, bvh->base(),
               bvh->prim_num()){};

    CPT_GPU LinearNode(float4 p1, float4 p2) {
        FLOAT4(aabb.mini) = p1;
        FLOAT4(aabb.maxi) = p2;
    }

  public:
    // The linearized BVH tree should contain: bound, base, prim_cnt,
    // rchild_offset, total_offset (to skip the entire node)
    AABB aabb;

    CPT_CPU_GPU_INLINE void get_range(int &beg, int &end) const noexcept {
        beg = aabb.base();
        end = aabb.prim_cnt();
    }
};

/**
 * @brief Compact BVH node (only 4 floats) for bandwidth reduction
 */
class CompactNode {
  private:
    static constexpr uint32_t LOW_8_MASK = 0xFF;         // 0000...1111 1111
    static constexpr uint32_t HIGH_24_MASK = 0xFFFFFF00; // 1111...0000 0000
    static constexpr uint32_t LOW_24_MASK = 0x00FFFFFF;  // 0000 1111...1111
    static constexpr uint32_t HIGH_SHIFT = 8;
    uint4 data;
    /**
     * The last uint32: high 24 bits | low 8 bits represents different meanings
     */
  public:
    CPT_CPU CompactNode(const float4 &front, const float4 &back) {
        HALF2(data.x) = Vec2Half(front.x, back.x);
        HALF2(data.y) = Vec2Half(front.y, back.y);
        HALF2(data.z) = Vec2Half(front.z, back.z);
        set_high_24bits(UINT_CREF_CAST(back.w));
        set_low_8bits(1);
    }

    CPT_GPU CompactNode() : data({0, 0, 0, 0}) {}
    CPT_GPU CompactNode(uint4 _data) : data(std::move(_data)) {}

    // set high 24 bits (signed)
    CPT_CPU void set_high_24bits(uint32_t val) {
        // clear the high 24 bits
        data.w &= LOW_8_MASK;

        // store as uint32
        uint32_t unsigned_val = val & LOW_24_MASK; // 24 bits
        data.w |= (unsigned_val << HIGH_SHIFT);
    }

    CPT_CPU void set_low_8bits(uint32_t val) {
        // clear low 8 bits
        data.w &= HIGH_24_MASK;
        data.w |= (val & LOW_8_MASK);
    }
    // unsigned 24 bits (upper bound: ~33M)
    CPT_GPU_INLINE uint32_t get_gmem_index() const noexcept {
        return (data.w >> HIGH_SHIFT) & LOW_24_MASK;
    }

    // unsigned 6 bits (upper bound: 63)
    CPT_GPU_INLINE uint32_t get_cached_offset() const noexcept {
        return data.w & LOW_8_MASK;
    }

    CPT_GPU_INLINE void unpack(Vec3 &mini, Vec3 &maxi) const {
        auto temp = __half22float2(CONST_HALF2(data.x));
        mini.x() = temp.x;
        maxi.x() = temp.y;
        temp = __half22float2(CONST_HALF2(data.y));
        mini.y() = temp.x;
        maxi.y() = temp.y;
        temp = __half22float2(CONST_HALF2(data.z));
        mini.z() = temp.x;
        maxi.z() = temp.y;
    }

    CPT_GPU bool intersect(Vec3 inv_d, Vec3 o_div, float &t_near) const {
        Vec3 mini, maxi;
        unpack(mini, maxi);
        auto t1s = mini.fmsub(inv_d, o_div);
        inv_d = maxi.fmsub(inv_d, o_div);

        float tmax = 0;
        t1s.min_max(inv_d, t_near, tmax);
        return (tmax > t_near) && (tmax > 0); // local memory access problem
    }
};

struct PrimMappingInfo {
    int obj_id;
    int prim_id;
    bool is_sphere;
    PrimMappingInfo() : obj_id(0), prim_id(0), is_sphere(false) {}
    PrimMappingInfo(int _obj_id, int _prim_id, bool _is_sphere)
        : obj_id(_obj_id), prim_id(_prim_id), is_sphere(_is_sphere) {}
};

struct AxisBins {
    AABB bound;
    int prim_cnt;

    AxisBins() : bound(1e5f, -1e5f, 0, 0), prim_cnt(0) {}

    void push(const BVHInfo &bvh) {
        bound += bvh.bound;
        prim_cnt++;
    }
};

void index_input(const std::vector<ObjInfo> &objs,
                 const std::vector<bool> &sphere_flags,
                 std::vector<PrimMappingInfo> &idxs, size_t num_primitives);

inline int object_index_packing(int obj_med_idx, int obj_id, bool is_sphere);

void create_bvh_info(const std::vector<Vec3> &points1,
                     const std::vector<Vec3> &points2,
                     const std::vector<Vec3> &points3,
                     const std::vector<PrimMappingInfo> &idxs,
                     const std::vector<int> &obj_med_idxs,
                     std::vector<BVHInfo> &bvh_infos);

void bvh_build(const std::vector<Vec3> &points1,
               const std::vector<Vec3> &points2,
               const std::vector<Vec3> &points3,
               const std::vector<ObjInfo> &objects,
               const std::vector<int> &obj_med_idxs,
               const std::vector<bool> &sphere_flags, const Vec3 &world_min,
               const Vec3 &world_max, std::vector<int> &obj_idxs,
               std::vector<int> &prim_idxs, std::vector<float4> &nodes,
               std::vector<CompactNode> &cache_nodes, int &max_cache_level,
               const int max_node_num, const float overlap_w);
