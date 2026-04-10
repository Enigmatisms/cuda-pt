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
 * @brief ReSTIR DI Spatial Reuse Kernel
 * @date: 2025.4.10
 *
 * After the initial RIS sampling + temporal reuse in the main kernel,
 * this kernel performs spatial reuse by merging neighboring pixels'
 * reservoirs with Jacobian correction via restir_compute_h().
 */
#include "renderer/megakernel_restir.cuh"

static constexpr int SPATIAL_NEIGHBOR_COUNT = 4;

CPT_KERNEL void restir_spatial_reuse_kernel(
    MediumPtrArray media, const ReSTIRReservoir *__restrict__ input_reservoirs,
    const ReSTIRGBuffer *__restrict__ g_buffer,
    ReSTIRReservoir *__restrict__ output_reservoirs, int img_w, int img_h,
    int envmap_id, int num_spatial_passes, int seed_offset) {

    // Neighbor offsets: right, left, up, down
    const int nx_off[SPATIAL_NEIGHBOR_COUNT] = {1, -1, 0, 0};
    const int ny_off[SPATIAL_NEIGHBOR_COUNT] = {0, 0, 1, -1};

    int px = threadIdx.x + blockIdx.x * blockDim.x,
        py = threadIdx.y + blockIdx.y * blockDim.y;
    if (px >= img_w || py >= img_h)
        return;

    int pix_idx = px + py * img_w;
    Sampler sampler(pix_idx, seed_offset);

    ReSTIRReservoir reservoir = input_reservoirs[pix_idx];
    const ReSTIRGBuffer &gb = g_buffer[pix_idx];

    if (!gb.is_valid || reservoir.w_sum <= 0.f) {
        output_reservoirs[pix_idx] = reservoir;
        return;
    }

    for (int pass = 0; pass < num_spatial_passes; pass++) {
        // Pick a random neighbor
        int neighbor_id = sampler.discrete1D() % SPATIAL_NEIGHBOR_COUNT;
        int nx = px + nx_off[neighbor_id];
        int ny = py + ny_off[neighbor_id];

        if (nx < 0 || nx >= img_w || ny < 0 || ny >= img_h)
            continue;

        int n_idx = nx + ny * img_w;
        const ReSTIRReservoir &neighbor = input_reservoirs[n_idx];
        const ReSTIRGBuffer &neighbor_gb = g_buffer[n_idx];

        if (!neighbor_gb.is_valid || neighbor.M <= 0 ||
            neighbor.sample.emitter_idx <= 0)
            continue;

        // Reevaluate neighbor's sample at current pixel's GBuffer
        Vec4 h_at_cur = restir_compute_h(neighbor.sample, gb, media, envmap_id);
        float w_at_cur = restir_luminance(h_at_cur);

        // Also reevaluate current sample at neighbor's GBuffer
        // (for symmetric pairwise MIS — simplified version skips this)
        restir_merge_reservoirs(reservoir, neighbor, w_at_cur,
                                sampler.next1D());
    }

    output_reservoirs[pix_idx] = reservoir;
}
