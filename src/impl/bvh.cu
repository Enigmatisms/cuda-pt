/**
 * @file bvh.cpp
 * @author Qianyue He
 * @date 2023.5 -> 2024.9
 * @brief BVH construction main logic
 * @copyright Copyright (c) 2023-2024
 */

#include <algorithm>
#include <numeric>
#include <array>
#include "core/bvh_opt.cuh"

struct PrimMappingInfo {
    int obj_id;
    int prim_id;
    bool is_sphere;
    PrimMappingInfo(): obj_id(0), prim_id(0), is_sphere(false) {}
    PrimMappingInfo(int _obj_id, int _prim_id, bool _is_sphere): obj_id(_obj_id), prim_id(_prim_id), is_sphere(_is_sphere) {}
};

struct AxisBins {
    AABB bound;
    int prim_cnt;

    AxisBins(): bound(1e5f, -1e5f, 0, 0), prim_cnt(0) {}

    void push(const BVHInfo& bvh) {
        bound += bvh.bound;
        prim_cnt ++;
    }
};

static constexpr int num_bins = 16;
static constexpr int no_div_threshold = 2;
static constexpr int sah_split_threshold = 8;
// A cluster with all the primitive centroid within a small range [less than 1e-3]
// is ill-posed. If there is more than 64 primitives, the primitives will be discarded
static constexpr float traverse_cost = 0.2f;

static float bvh_overlap_w = 1.f;
static int max_depth = 0;

SplitAxis BVHNode::max_extent_axis(const std::vector<BVHInfo>& bvhs, float& min_r, float& interval) const {
    int _base = base(), _prim_num = prim_num();
    Vec3 min_ctr = bvhs[_base].centroid, max_ctr = bvhs[_base].centroid;
    for (int i = 1; i < _prim_num; i++) {
        Vec3 ctr = bvhs[_base + i].centroid;
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

void index_input(
    const std::vector<ObjInfo>& objs, 
    const std::vector<bool>& sphere_flags, 
    std::vector<PrimMappingInfo>& idxs, size_t num_primitives
) {
    // input follow the shape of the number of objects, for each position
    // the number of primitive / whether the primitive is sphere will be stored, the index will be object id
    size_t result_shape = objs.size();      // shape is (3, obj_num)
    idxs.reserve(num_primitives);                   // accumulate(num_ptr, num_ptr + result_shape) = num_primitives
    int prim_num = 0;
    for (size_t i = 0; i < result_shape; i++) {
        int local_num = objs[i].prim_num;
        int obj_id = static_cast<int>(i);
        for (int j = 0; j < local_num; j++) {
            idxs.emplace_back(obj_id, j + prim_num, sphere_flags[i]);
        }
        prim_num += local_num;
    }
}

inline int object_index_packing(int obj_med_idx, int obj_id, bool is_sphere) {
    // take the lower 20 bits and shift up 20bits
    int trucated = (obj_med_idx & 0x00000fff) << 20;
    return (static_cast<int>(is_sphere) << 31) + trucated + (obj_id & 0x000fffff);
}

void create_bvh_info(
    const std::vector<Vec3>& points1,
    const std::vector<Vec3>& points2,
    const std::vector<Vec3>& points3,
    const std::vector<PrimMappingInfo>& idxs, 
    const std::vector<int>& obj_med_idxs,        // encodes both medium mapping and alpha cull-able info
    std::vector<BVHInfo>& bvh_infos
) {
    bvh_infos.reserve(points1.size());
    for (size_t i = 0; i < points1.size(); i++) {
        const auto& idx_info = idxs[i];
        // idx_info.first is the primitive_id, while second is obj_id (negative means)
        const int obj_med_id = obj_med_idxs[idx_info.obj_id];
        bvh_infos.emplace_back(points1[i], points2[i], points3[i], 
            object_index_packing(obj_med_id, idx_info.obj_id, idx_info.is_sphere), 
            idx_info.prim_id, idx_info.is_sphere);
    }
}

int recursive_bvh_SAH(BVHNode* const cur_node, std::vector<BVHInfo>& bvh_infos, int depth = 0, int max_prim_node = 16) {
    AABB fwd_bound(1e5f, -1e5f, 0, 0), bwd_bound(1e5f, -1e5f, 0, 0);
    int child_prim_cnt = 0;                // this index is used for indexing variable `bins`
    const int prim_num = cur_node->prim_num(), base = cur_node->base(), max_pos = base + prim_num;
    float min_cost = 5e9f, node_prim_cnt = float(prim_num);
    
    // Step 1: decide the axis that expands the maximum extent of space
    float min_range = 0, interval = 0;
    SplitAxis max_axis = cur_node->max_extent_axis(bvh_infos, min_range, interval);
    
    if (max_axis != SplitAxis::AXIS_NONE && prim_num > sah_split_threshold) {   // SAH
        // Step 2: binning the space
        std::array<AxisBins, num_bins> idx_bins;
        for (int i = base; i < max_pos; i++) {
            int index = std::min((int)floorf((bvh_infos[i].centroid[max_axis] - min_range) / interval), num_bins - 1);
            idx_bins[index].push(bvh_infos[i]);
        }

        // Step 3: forward-backward linear sweep for heuristic calculation
        std::array<int, num_bins> prim_cnts;
        std::array<float, num_bins> fwd_areas, bwd_areas;
        std::vector<AABB> fwd_aabbs, bwd_aabbs;     // to calculate AABB intersection
        fwd_aabbs.reserve(num_bins);
        bwd_aabbs.reserve(num_bins);
        prim_cnts.fill(0);
        fwd_areas.fill(0);
        bwd_areas.fill(0);
        for (int i = 0; i < num_bins; i++) {
            fwd_bound   += idx_bins[i].bound;
            prim_cnts[i] = idx_bins[i].prim_cnt;
            fwd_areas[i] = fwd_bound.area();
            fwd_aabbs.push_back(fwd_bound);
            if (i > 0) {
                bwd_bound += idx_bins[num_bins - i].bound;
                bwd_areas[num_bins - 1 - i] = bwd_bound.area();
                bwd_aabbs.push_back(bwd_bound);
            }
        }
        cur_node->bound.mini = fwd_bound.mini;
        cur_node->bound.maxi = fwd_bound.maxi;
        float node_inv_area = 1. / cur_node->bound.area();
        std::partial_sum(prim_cnts.begin(), prim_cnts.end(), prim_cnts.begin());

        // Step 4: use the calculated area to computed the segment boundary
        int seg_bin_idx = 0;
        for (int i = 0; i < num_bins - 1; i++) {
            float intrsct_a = fwd_aabbs[i].intersection_area(bwd_aabbs.back());
            float cost = traverse_cost + node_inv_area * (
                intrsct_a * std::max(bvh_overlap_w - 0.5f, 0.f) * node_prim_cnt +
                float(prim_cnts[i]) * fwd_areas[i] + (node_prim_cnt - float(prim_cnts[i])) * bwd_areas[i]
            );
            if (cost < min_cost) {
                min_cost = cost;
                seg_bin_idx = i;
            }
            bwd_aabbs.pop_back();
        }
        // Step 5: reordering the BVH info in the vector to make the segment contiguous (partition around pivot)
        if (min_cost < node_prim_cnt || prim_num > max_prim_node) {
            std::partition(bvh_infos.begin() + base, bvh_infos.begin() + max_pos,
                [pivot = min_range + interval * float(seg_bin_idx + 1), dim = max_axis](const BVHInfo& bvh) {
                    return bvh.centroid[dim] < pivot;
            });
            child_prim_cnt = prim_cnts[seg_bin_idx];
        }

        fwd_bound.clear();
        bwd_bound.clear();
        for (int i = 0; i <= seg_bin_idx; i++)       // calculate child node bound
            fwd_bound += idx_bins[i].bound;
        for (int i = seg_bin_idx + 1; i < num_bins; i++)
            bwd_bound += idx_bins[i].bound;
    } else {                                    // equal primitive number split (two nodes have identical primitives)
        int seg_idx = (base + max_pos) >> 1;
        // Step 5: reordering the BVH info in the vector to make the segment contiguous (keep around half of the bvh in lchild)
        if (max_axis != SplitAxis::AXIS_NONE) {
            std::nth_element(bvh_infos.begin() + base, bvh_infos.begin() + seg_idx, bvh_infos.begin() + max_pos,
                [dim = max_axis] (const BVHInfo& bvh1, const BVHInfo& bvh2) {
                    return bvh1.centroid[dim] < bvh2.centroid[dim];
                }
            );
        }
        for (int i = base; i < seg_idx; i++)    // calculate child node bound
            fwd_bound += bvh_infos[i].bound;
        for (int i = seg_idx; i < max_pos; i++)
            bwd_bound += bvh_infos[i].bound;
        cur_node->bound += fwd_bound;
        cur_node->bound += bwd_bound;
        child_prim_cnt = seg_idx - base;        // bvh[seg_idx] will be in rchild
        float intrsct_a = fwd_bound.intersection_area(bwd_bound);
        float split_cost = traverse_cost + (1.f / cur_node->bound.area()) * (
            intrsct_a * std::max(bvh_overlap_w - 0.5f, 0.f) * node_prim_cnt + \
            fwd_bound.area() * float(child_prim_cnt) + bwd_bound.area() * (node_prim_cnt - float(child_prim_cnt))
        );
        if (split_cost >= node_prim_cnt && prim_num < max_prim_node)
            child_prim_cnt = 0;
    }

    if (child_prim_cnt > 0) {             // cost of splitting is less than making this node a leaf node
        // Step 5: split the node and initialize the children
        cur_node->lchild = new BVHNode(base, child_prim_cnt);
        cur_node->rchild = new BVHNode(base + child_prim_cnt, prim_num - child_prim_cnt);

        cur_node->lchild->bound.mini = fwd_bound.mini;
        cur_node->lchild->bound.maxi = fwd_bound.maxi;
        cur_node->rchild->bound.mini = bwd_bound.mini;
        cur_node->rchild->bound.maxi = bwd_bound.maxi;
        cur_node->axis = max_axis;
        // Step 7: start recursive splitting for the children
        int node_num = 1;
        if (cur_node->lchild->prim_num() > no_div_threshold)
            node_num += recursive_bvh_SAH(cur_node->lchild, bvh_infos, depth + 1, max_prim_node);
        else {
            max_depth = std::max(depth + 1, max_depth);
            node_num ++;
        }
        if (cur_node->rchild->prim_num() > no_div_threshold)
            node_num += recursive_bvh_SAH(cur_node->rchild, bvh_infos, depth + 1, max_prim_node);
        else {
            max_depth = std::max(depth + 1, max_depth);
            node_num ++;
        }
        return node_num;
    } else {
        // This is a leaf node, yet this is the only way that a leaf node contains more than one primitive
        cur_node->axis = AXIS_NONE;
        max_depth = std::max(depth, max_depth);
        return 1;
    }
}

static BVHNode* bvh_root_start(const Vec3& world_min, const Vec3& world_max, int& node_num, std::vector<BVHInfo>& bvh_infos, int max_prim_node = 16) {
    // Build BVH tree root node and start recursive tree construction
    printf("[BVH] World min: ");
    print_vec3(world_min);
    printf("[BVH] World max: ");
    print_vec3(world_max);
    BVHNode* root_node = new BVHNode(0, bvh_infos.size());
    root_node->bound.mini = world_min;
    root_node->bound.maxi = world_max;
    node_num = recursive_bvh_SAH(root_node, bvh_infos, max_prim_node);
    return root_node;
}

// This is the final function call for `bvh_build`
static int recursive_linearize(
    BVHNode* cur_node, 
    std::vector<float4>& nodes,
    std::vector<CompactNode>& cache_nodes,
    const int depth = 0,
    const int cache_max_depth = 4
) {
    // BVH tree should be linearized to better traverse and fit in the system memory
    // The linearized BVH tree should contain: bound, base, prim_cnt, rchild_offset, total_offset (to skip the entire node)
    // Note that if rchild_offset is -1, then the node is leaf. Leaf node points to primitive array
    // which is already sorted during BVH construction, containing primitive_id and obj_id for true intersection
    // Note that lin_nodes has been reserved
    size_t current_size = nodes.size() >> 1, current_cached = cache_nodes.size();
    float4 node_f, node_b;
    cur_node->get_float4(node_f, node_b);
    nodes.push_back(node_f);
    nodes.push_back(node_b);
    reinterpret_cast<uint32_t&>(node_f.w) = 1;               // always assume leaf node (offset = 1)
    reinterpret_cast<uint32_t&>(node_b.w) = current_size;
    if (depth < cache_max_depth) {
        // LinearNode (cached): 
        // (float3) aabb.min
        // (int)    jump offset to next cached node 
        // (float3) aabb.max 
        // (int)    index to the global memory node (if -1, means it it not a leave node, we should continue)
        cache_nodes.emplace_back(node_f, node_b);
    }
    /**
     * @note
     * Clarify on how do we store BVH range and node offsets:
     * - for non-leaf nodes, since beg_idx and end_idx will not be used, we only need node_offset
     *   SO node_offset is stored as the `NEGATIVE` value, so if we encounter a negative float4.w, we know
     *   that the current node is non-leaf
     * - for leaf nodes, we don't modify the float4.w
     */
    if (cur_node->lchild != nullptr) {
        // non-leaf node
        int lnodes = recursive_linearize(
            cur_node->lchild, nodes, 
            cache_nodes, depth + 1, cache_max_depth
        );
        lnodes += recursive_linearize(
            cur_node->rchild, nodes,
            cache_nodes, depth + 1, cache_max_depth
        );
        INT_REF_CAST(nodes[2 * current_size + 1].w) = -(lnodes + 1);
        if (depth < cache_max_depth) {
            // store the jump offset to the next cached node (for non-leaf node)
            cache_nodes[current_cached].set_low_8bits(cache_nodes.size() - current_cached);
        }
        return lnodes + 1;                      // include the cur_node                       
    } else {
        // leaf node has negative offset
        return 1;
    }
}

// Try to use two threads to build the BVH
void bvh_build(
    const std::vector<Vec3>& points1,
    const std::vector<Vec3>& points2,
    const std::vector<Vec3>& points3,
    const std::vector<ObjInfo>& objects,
    const std::vector<int>& obj_med_idxs,
    const std::vector<bool>& sphere_flags,
    const Vec3& world_min, const Vec3& world_max,
    std::vector<int>& obj_idxs, 
    std::vector<int>& prim_idxs, 
    std::vector<float4>& nodes,
    std::vector<CompactNode>& cache_nodes,
    int& cache_max_level,
    const int max_prim_node,
    const float overlap_w
) {
    bvh_overlap_w = overlap_w;
    std::vector<PrimMappingInfo> idx_prs;
    std::vector<BVHInfo> bvh_infos;
    int node_num = 0, num_prims_all = points1.size();
    index_input(objects, sphere_flags, idx_prs, num_prims_all);
    create_bvh_info(points1, points2, points3, idx_prs, obj_med_idxs, bvh_infos);
    BVHNode* root_node = bvh_root_start(world_min, world_max, node_num, bvh_infos, max_prim_node);
    float total_cost = calculate_cost(root_node, traverse_cost);
    
    printf("[BVH] BVH tree max depth: %d\n", max_depth);
    printf("[BVH] Traversed BVH SAH cost: %.7f, AVG: %.7f\n", total_cost, total_cost / static_cast<float>(bvh_infos.size()));
    cache_max_level = std::min(std::max(max_depth - 1, 0), cache_max_level);
    nodes.reserve(node_num << 1);
    cache_nodes.reserve(1 << cache_max_level);
    recursive_linearize(root_node, nodes, cache_nodes, 0, cache_max_level);
    printf("[BVH] Number of nodes to cache: %llu (%d)\n", cache_nodes.size(), cache_max_level);

    // FIXME: MASK ALPHA, change obj_idxs
    obj_idxs.reserve(bvh_infos.size());
    prim_idxs.reserve(bvh_infos.size());
    for (BVHInfo& bvh: bvh_infos) {
        obj_idxs.emplace_back(bvh.bound.__bytes1);
        prim_idxs.emplace_back(bvh.bound.__bytes2);
    }
    delete root_node;
}