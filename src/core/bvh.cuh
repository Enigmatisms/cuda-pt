/**
 * @file bvh.h
 * @author Qianyue He
 * @brief BVH utilities
 * @copyright Copyright (c) 2023-2024
 */

#pragma once
#include <algorithm>
#include "core/object.cuh"
#include "core/vec2_half.cuh"
#include "core/constants.cuh"

enum SplitAxis: int {AXIS_X, AXIS_Y, AXIS_Z, AXIS_NONE};

struct BVHInfo {
    // BVH is for both triangle meshes and spheres
    AABB bound;
    Vec3 centroid;

    BVHInfo(): bound(1e5f, -1e5f, 0, 0), centroid() {}
    BVHInfo(const Vec3& p1, const Vec3& p2, const Vec3& p3, 
            int obj_idx, int prim_idx, bool is_sphere = false)
    {
        // Extract two vertices for the primitive, once converted to AABB
        // We don't need to distinguish between mesh or sphere
        // Note that vertex vectors in the primitive matrix are col-major order
        if (is_sphere) {
            centroid = p1;
            bound    = AABB(p1 - p2.x() - AABB_EPS, p1 + p2.x() + AABB_EPS, obj_idx, prim_idx);
        } else {
            centroid = (p1 + p2 + p3) * 0.33333333333f;      // barycenter
            bound    = AABB(p1, p2, p3, obj_idx, prim_idx);
        }
        auto range = bound.range();
    }

    void get_float4(float4& node_f, float4& node_b) const {
        node_f.x = bound.mini.x();
        node_f.y = bound.mini.y();
        node_f.z = bound.mini.z();
        node_f.w = *reinterpret_cast<const float*>(&bound.__bytes1);
        node_b.x = bound.maxi.x();
        node_b.y = bound.maxi.y();
        node_b.z = bound.maxi.z();
        node_b.w = *reinterpret_cast<const float*>(&bound.__bytes2);
    }
};

class BVHNode {
public:
    BVHNode(): bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, 0, 0), axis(AXIS_NONE), lchild(nullptr), rchild(nullptr) {}
    BVHNode(int base, int prim_num): 
        bound(AABB_INVALID_DIST, -AABB_INVALID_DIST, base, prim_num), axis(AXIS_NONE), lchild(nullptr), rchild(nullptr) {}
    ~BVHNode() {
        if (lchild != nullptr) delete lchild;
        if (rchild != nullptr) delete rchild;
    }

    bool is_leaf() const { return lchild == nullptr; }

    int base() const { return bound.base(); }
    int& base() { return bound.base(); }

    int prim_num() const { return bound.prim_cnt(); }
    int& prim_num() { return bound.prim_cnt(); }

    void get_float4(float4& node_f, float4& node_b) const {
        node_f.x = bound.mini.x();
        node_f.y = bound.mini.y();
        node_f.z = bound.mini.z();
        node_f.w = *reinterpret_cast<const float*>(&bound.__bytes1);
        node_b.x = bound.maxi.x();
        node_b.y = bound.maxi.y();
        node_b.z = bound.maxi.z();
        node_b.w = *reinterpret_cast<const float*>(&bound.__bytes2);
    }
public:
    // The axis start and end are scaled up a little bit
    SplitAxis max_extent_axis(const std::vector<BVHInfo>& bvhs, float& min_r, float& interval) const;
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
 * neither can we record whether the node is accessed nor store the node to be accessed
 * in a stack. Allocate a global field which might incur (H * W * D * 4) B memory consumption,
 * where D is the depth of BVH tree, which can be extremely inbalanced when the primitives
 * are distributed with poor uniformity. Therefore, in this implementation, we can only opt
 * for a suboptimal solution, to traverse the tree using just DFS, resulting in a simple
 * traversal method: the index for the linear node container is monotonously increasing. So 
 * in this implementation, split axis will not be included (which will not be used anyway).
 * TODO: we can opt for dynamic snode, but I think this would be ugly. Simple unidirectional traversal
 * would be much better than brute-force traversal with only AABB pruning.
 * 
 * This is actually borrowed from my AdaPT repo. Now, for CUDA, things are the same:
 * dynamically indexing a local array will have the data winding up in the GMEM, which can be slow
 * So, I will still implement a jump-back free version, with no stack 
 */
class LinearNode {
public:
    CPT_CPU_GPU LinearNode(): aabb(1e5f, -1e5f, 0, 0) {}

    // linear nodes are initialized during DFS binary tree traversal
    CPT_CPU LinearNode(const BVHNode *const bvh): aabb(bvh->bound.mini, bvh->bound.maxi, bvh->base(), bvh->prim_num()) {};       

    CPT_GPU LinearNode(float4 p1, float4 p2) {
        FLOAT4(aabb.mini) = p1;
        FLOAT4(aabb.maxi) = p2;
    }
public:
    // The linearized BVH tree should contain: bound, base, prim_cnt, rchild_offset, total_offset (to skip the entire node)
    AABB aabb;

    CPT_CPU_GPU_INLINE void get_range(int& beg, int& end) const noexcept {
        beg = aabb.base();
        end = aabb.prim_cnt();
    }
};

/**
 * @brief Compact BVH node (only 4 floats) for bandwidth reduction
 */
class CompactNode {
private:
    static constexpr uint32_t LOW_6_MASK = 0x3F;           // 000...0011 1111
    static constexpr uint32_t HIGH_26_MASK = 0xFFFFFFC0;  // 111...1100 0000
    static constexpr uint32_t LOW_26_MASK = 0x03FFFFFF;  // 0000 0011...1111
    static constexpr int HIGH_SHIFT = 6;
    uint4 data;
    /**
     * The last uint32: high 26 bits | low 6 bits represents different meanings
     */
public:
    CPT_CPU CompactNode(const float4& front, const float4& back, bool not_leaf = true, bool smem_cached = false) {
        HALF2(data.x) = Vec2Half(front.x, back.x);
        HALF2(data.y) = Vec2Half(front.y, back.y);
        HALF2(data.z) = Vec2Half(front.z, back.z);
        if (smem_cached) {
            set_high_26bits(INT_CREF_CAST(back.w));
            set_low_6bits(1);
        } else {
            // for non-leaf nodes, beg_idx is useless, node_num is used to skip over the sub-tree
            // for leaf nodes, beg_idx and node_num is both useful, with node_num no more than 31
            // non-leaf will node have its `beg_idx` set in the constructor
            // instead, it will be set in `recursive_linearize` with -(lnodes + 1), which is negative
            // so, negative beg_idx indicates non-leaf nodes
            int beg_idx = INT_CREF_CAST(front.w), node_num = INT_CREF_CAST(back.w);
            if (not_leaf) {
                set_low_6bits(0);
            } else {
                set_high_26bits(beg_idx);
                set_low_6bits(node_num);
            }
        }
    }

    CPT_GPU CompactNode(): data({0, 0, 0, 0}) {}
    CPT_GPU CompactNode(uint4 _data): data(std::move(_data)) {}

    // set high 26 bits (signed)
    CPT_CPU void set_high_26bits(int val) {
        // clear the high 26 bits
        data.w &= LOW_6_MASK;

        // store as uint32
        uint32_t unsigned_val = static_cast<uint32_t>(val) & LOW_26_MASK; // 26 bits
        data.w |= (unsigned_val << HIGH_SHIFT);
    }

    CPT_CPU void set_low_6bits(int val) {
        // clear low 6 bits
        data.w &= HIGH_26_MASK;
        data.w |= (val & LOW_6_MASK);
    }
    // signed 26 bits (upper bound: ~33M)
    CPT_GPU_INLINE int get_gmem_index() const noexcept {
        uint32_t high = (data.w >> HIGH_SHIFT) & LOW_26_MASK;
        // place the sign bit at bit 31, then perform an arithmetic right shift for sign-extend
        return (static_cast<int>(high << HIGH_SHIFT)) >> HIGH_SHIFT;
    }

    // signed 26 bits (upper bound: ~33M)
    CPT_GPU_INLINE int get_beg_idx() const noexcept {
        uint32_t high = (data.w >> HIGH_SHIFT) & LOW_26_MASK;
        // place the sign bit at bit 31, then perform an arithmetic right shift for sign-extend
        return (static_cast<int>(high << HIGH_SHIFT)) >> HIGH_SHIFT;
    }

    // unsigned 6 bits (upper bound: 63)
    CPT_GPU_INLINE uint32_t get_cached_offset() const noexcept {
        return data.w & LOW_6_MASK;
    }

    // unsigned 6 bits (upper bound: 63)
    CPT_GPU_INLINE uint32_t get_prim_cnt() const noexcept {
        return data.w & LOW_6_MASK;
    }

    CPT_GPU_INLINE void unpack(Vec3& mini, Vec3& maxi) const {
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

    CPT_GPU bool intersect(Vec3 inv_d, Vec3 o_div, float& t_near) const {
        Vec3 mini, maxi;
        unpack(mini, maxi);
        auto t1s = mini.fmsub(inv_d, o_div);
        inv_d    = maxi.fmsub(inv_d, o_div);

        float tmax = 0;
        t1s.min_max(inv_d, t_near, tmax);
        return (tmax > t_near) && (tmax > 0);             // local memory access problem
    }
};

void bvh_build(
    const std::vector<Vec3>& points1,
    const std::vector<Vec3>& points2,
    const std::vector<Vec3>& points3,
    const std::vector<ObjInfo>& objects,
    const std::vector<bool>& sphere_flags,
    const Vec3& world_min, const Vec3& world_max,
    std::vector<int>& obj_idxs, 
    std::vector<int>& prim_idxs, 
    std::vector<CompactNode>& nodes,
    std::vector<CompactNode>& cached_nodes,
    int& max_cache_level,
    const int max_node_num,
    const float overlap_w
);
