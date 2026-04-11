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
 * In-place spatial reuse: reads and writes the same reservoirs buffer.
 * Each thread loads its own reservoir into registers, merges with
 * neighbors (benign read-after-write on neighbor pixels is acceptable),
 * then writes back only its own pixel.
 */
#include "renderer/megakernel_restir.cuh"

CPT_KERNEL void restir_spatial_reuse_kernel(
    MediumPtrArray media, ReSTIRReservoir *__restrict__ reservoirs,
    const ReSTIRGBuffer *__restrict__ g_buffer, int img_w, int img_h,
    int envmap_id, int num_spatial_passes, int seed_offset) {

    int px = threadIdx.x + blockIdx.x * blockDim.x,
        py = threadIdx.y + blockIdx.y * blockDim.y;
    if (px >= img_w || py >= img_h)
        return;

    int pix_idx = px + py * img_w;
    Sampler sampler(pix_idx, seed_offset);

    ReSTIRReservoir reservoir = reservoirs[pix_idx];
    const ReSTIRGBuffer &gb = g_buffer[pix_idx];

    if (!gb.is_valid || reservoir.w_sum <= 0.f) {
        return;
    }

    for (int pass = 0; pass < num_spatial_passes; pass++) {
        // Pick a random neighbor (4-connected)
        int off = sampler.discrete1D() & 3;
        int nx = px + (1 - (off & 1) * 2) * (1 - (off >> 1));
        int ny = py + (1 - (off & 1) * 2) * (off >> 1);

        if (nx < 0 || nx >= img_w || ny < 0 || ny >= img_h)
            continue;

        int n_idx = nx + ny * img_w;
        const ReSTIRReservoir &neighbor = reservoirs[n_idx];
        const ReSTIRGBuffer &neighbor_gb = g_buffer[n_idx];

        if (!neighbor_gb.is_valid || neighbor.M <= 0 ||
            neighbor.sample.emitter_idx <= 0)
            continue;

        // Reevaluate neighbor's sample at current pixel's GBuffer
        Vec4 h_at_cur = restir_compute_h(neighbor.sample, gb, media, envmap_id);
        float w_at_cur = restir_luminance(h_at_cur);

        restir_merge_reservoirs(reservoir, neighbor, w_at_cur,
                                sampler.next1D());
    }

    reservoirs[pix_idx] = reservoir;
}
