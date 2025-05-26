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

// TODO(heqianyue): rewrite range-based non-repeating contiguous BVH recursive
// build as the SBVH node build. I believe it can also be exported as Linearized
// nodes even, SMEM node caching can be applied
