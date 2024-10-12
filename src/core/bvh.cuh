/**
 * @file bvh.h
 * @author Qianyue He
 * @brief BVH utilities
 * @copyright Copyright (c) 2023-2024
 */

#pragma once
#include <algorithm>
#include "core/object.cuh"
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
    BVHNode(): bound(1e5f, -1e5f, 0, 0), axis(AXIS_NONE), lchild(nullptr), rchild(nullptr) {}
    BVHNode(int base, int prim_num): bound(1e5f, -1e5f, base, prim_num), axis(AXIS_NONE), lchild(nullptr), rchild(nullptr) {}
    ~BVHNode() {
        if (lchild != nullptr) delete lchild;
        if (rchild != nullptr) delete rchild;
    }

    CPT_CPU_GPU int base() const { return bound.base(); }
    CPT_CPU_GPU int& base() { return bound.base(); }

    CPT_CPU_GPU int prim_num() const { return bound.prim_cnt(); }
    CPT_CPU_GPU int& prim_num() { return bound.prim_cnt(); }

    CPT_CPU void get_float4(float4& node_f, float4& node_b) const {
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
    SplitAxis max_extent_axis(const std::vector<BVHInfo>& bvhs, std::vector<float>& bins) const;
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
    CPT_CPU_GPU LinearNode(const BVHNode *const bvh): aabb(bvh->bound.mini, bvh->bound.maxi, bvh->base(), bvh->prim_num()) {};       

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

    CPT_GPU_INLINE void export_aabb(LinearNode& node) const noexcept {
        node.aabb.copy_from(aabb);
    }
};

void bvh_build(
    const std::vector<Vec3>& points1,
    const std::vector<Vec3>& points2,
    const std::vector<Vec3>& points3,
    const std::vector<ObjInfo>& objects,
    const std::vector<bool>& sphere_flags,
    const Vec3& world_min, const Vec3& world_max,
    std::vector<int2>& bvh_nodes, 
    std::vector<float4>& node_fronts,
    std::vector<float4>& node_backs,
    std::vector<float4>& cache_fronts,
    std::vector<float4>& cache_backs,
    int& max_cache_level
);
