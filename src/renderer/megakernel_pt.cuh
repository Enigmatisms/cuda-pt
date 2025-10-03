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
 * @brief Megakernel Path Tracing
 * @date: 2024.9.15
 */
#pragma once
#include "core/camera_model.cuh"
#include "core/max_depth.h"
#include "renderer/scheduler.cuh"
#include "renderer/tracing_func.cuh"

CPT_CPU_INLINE int get_max_block() {
    cudaDeviceProp prop;
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "[Error] No viable device found, please check.\n";
        throw std::runtime_error("[No Device Error]");
    }
    cudaGetDeviceProperties(&prop, 0);
    return prop.multiProcessorCount;
}

#define LAUNCH_PT_KERNEL(scheduler_type, render_once, grid_size, block_size,   \
                         cached_size, ...)                                     \
    render_pt_kernel<scheduler_type, render_once>                              \
        <<<grid_size, block_size, cached_size>>>(__VA_ARGS__)

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level
 * culling shared memory might not be easy to use, since the memory granularity
 * will be too difficult to control
 *
 * @param objects        object encapsulation
 * @param verts          vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms          normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3,
 * 3D)
 * @param uvs            uv coordinates, Packed 3 Half2 and 1 int for padding
 * (sum up to 128 bits)
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
 * @param md_params      maximum allowed bounces (total, diffuse, specular,
 * transmission)
 * @param node_num       number of nodes on a BVH tree
 * @param accum_cnt      Counter of iterations
 * @param cache_num      Number of cached BVH nodes
 * @param gamma_corr     For online rendering, whether to enable gamma
 * correction on visualization
 */
template <typename Scheduler, bool render_once>
CPT_KERNEL void render_pt_kernel(
    const DeviceCamera &dev_cam, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects, ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, DeviceImage image, const MaxDepthParams md_params,
    float *__restrict__ output_buffer, float *__restrict__ var_buffer,
    int num_emitter, int seed_offset, int node_num = -1, int accum_cnt = 1,
    int cache_num = 0, int envmap_id = 0, bool gamma_corr = false);

/**
 * Megakernel Light Tracing. Light tracing is only used to render
 * complex caustics: starting from the emitter, we will only record
 * path which has more than `specular_constraints` number of
 * specular nodes
 * @param specular_constraints The path throughput will be ignored
 * if number of specular events is less or equal to this value
 */
template <bool render_once>
CPT_KERNEL void
render_lt_kernel(const DeviceCamera &dev_cam, const PrecomputedArray verts,
                 const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
                 ConstObjPtr objects, ConstIndexPtr emitter_prims,
                 const cudaTextureObject_t bvh_leaves,
                 const cudaTextureObject_t nodes, ConstF4Ptr cached_nodes,
                 DeviceImage image, const MaxDepthParams md_params,
                 float *__restrict__ output_buffer, int num_emitter,
                 int seed_offset, int node_num = -1, int accum_cnt = 1,
                 int cache_num = 0, int specular_constraints = 0,
                 float caustic_scale = 1.f, bool gamma_corr = false);
