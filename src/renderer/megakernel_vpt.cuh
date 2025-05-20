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
 * @author: Qianyue He
 * @brief Megakernel Volumetric Path Tracing
 *
 * Note that, the current implementation does not support nested volume that are
 * 4 or more levels. [volume-1  [volume-2 [volume-3]   ]   ]  is the limit
 * Actually, it won't trigger any fault but the rendering result will not be
 * correct for more layers than 3
 *
 * Also, non-strict nesting ( [vol-1    [vol-1 & vol-2 intersection]    vol-2])
 * will also be erroneous, it won't break down but the result will also be
 * incorrect.
 * @date: 2025.2.7
 */
#pragma once
#include "core/camera_model.cuh"
#include "core/max_depth.h"
#include "core/medium.cuh"
#include "renderer/tracing_func.cuh"
#include <cuda/pipeline>

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level
 * culling shared memory might not be easy to use, since the memory granularity
 * will be too difficult to control
 *
 * @param verts          vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms          normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3,
 * 3D)
 * @param uvs            uv coordinates, Packed 3 Half2 and 1 int for padding
 * (sum up to 128 bits)
 * @param objects        object encapsulation
 * @param media          Array of Medium base class pointers
 * @param emitter_prims  Primitive indices for emission objects
 * @param bvh_leaves     BVH leaf nodes (int texture, storing primitive to obj
 * index map)
 * @param nodes          BVH nodes (32 Bytes)
 * @param cached_nodes   BVH cached nodes (in shared memory): first half: front
 * float4, second half: back float4
 * @param image          GPU image buffer
 * @param output_buffer  Possible visualization buffer
 * @param num_emitter    number of emitters
 * @param seed_offset    offset to random seed (to create uncorrelated samples)
 * @param cam_vol_idx    If camera is inside the volume, the ray spawned will
 * have initial volume id to store
 * @param md_params      maximum allowed bounces (total, diffuse, specular,
 * transmission)
 * @param node_num       number of nodes on a BVH tree
 * @param accum_cnt      Counter of iterations
 * @param cache_num      Number of cached BVH nodes
 * @param gamma_corr     For online rendering, whether to enable gamma
 * correction on visualization
 */
template <bool render_once>
CPT_KERNEL void render_vpt_kernel(
    const DeviceCamera &dev_cam, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    MediumPtrArray media, ConstObjPtr objects, ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, DeviceImage image, const MaxDepthParams md_params,
    float *__restrict__ output_buffer, float *__restrict__ var_buffer,
    int num_emitter, int seed_offset, int cam_vol_idx = 0, int node_num = -1,
    int accum_cnt = 1, int cache_num = 0, int envmap_id = 0,
    bool gamma_corr = false);
