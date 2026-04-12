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
 * @brief Megakernel ReSTIR DI Path Tracing kernel declaration
 * @date: 2025.4.10
 */
#pragma once
#include "core/camera_model.cuh"
#include "core/max_depth.h"
#include "renderer/restir.cuh"
#include "renderer/scheduler.cuh"
#include "renderer/tracing_func.cuh"

#define LAUNCH_RESTIR_KERNEL(render_once, grid_size, block_size, cached_size,  \
                             ...)                                              \
    render_pt_restir_kernel<render_once>                                       \
        <<<grid_size, block_size, cached_size>>>(__VA_ARGS__)

/**
 * @brief ReSTIR DI Path Tracing kernel
 *
 * Combines surface + volumetric path tracing with ReSTIR direct illumination
 * at the first bounce. Subsequent bounces use classic NEE.
 *
 * @param media          Array of Medium pointers (volumetric support)
 * @param cam_vol_idx    Camera volume index (0 if camera not in a volume)
 * @param prev_reservoirs Previous frame reservoirs for temporal reuse
 * @param curr_reservoirs Current frame reservoirs (output)
 * @param g_buffer       G-buffer for ReSTIR reuse (output)
 * @param temporal_reuse Whether to enable temporal reuse
 * (other params same as render_pt_kernel, restir_m via md_params)
 */
template <bool render_once>
CPT_KERNEL void render_pt_restir_kernel(
    const DeviceCamera &dev_cam, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    MediumPtrArray media, ConstObjPtr objects, ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, DeviceImage image, const MaxDepthParams md_params,
    float *__restrict__ output_buffer, float *__restrict__ var_buffer,
    int num_emitter, int seed_offset, int cam_vol_idx, int node_num,
    int accum_cnt, int cache_num, int envmap_id, bool gamma_corr,
    const ReSTIRReservoir *__restrict__ prev_reservoirs,
    ReSTIRReservoir *__restrict__ curr_reservoirs,
    ReSTIRGBuffer *__restrict__ g_buffer, bool temporal_reuse,
    const ReSTIRGIReservoir *__restrict__ prev_gi_reservoirs,
    ReSTIRGIReservoir *__restrict__ curr_gi_reservoirs);
