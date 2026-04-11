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
 * @brief ReSTIR DI Path Tracer host implementation
 * @date: 2025.4.10
 */
#include "renderer/megakernel_restir.cuh"
#include "renderer/restir_pt.cuh"

// Forward declaration of spatial reuse kernel (defined in restir_spatial.cu)
CPT_KERNEL void restir_spatial_reuse_kernel(
    MediumPtrArray media, ReSTIRReservoir *__restrict__ reservoirs,
    const ReSTIRGBuffer *__restrict__ g_buffer, int img_w, int img_h,
    int envmap_id, int num_spatial_passes, int seed_offset);

static constexpr int SEED_SCALER = 11467;
static constexpr int SHFL_THREAD_X = 5;
static constexpr int SHFL_THREAD_Y = 2;
static constexpr int SPATIAL_PASSES = 1;

void ReSTIRPathTracer::allocate_restir_buffers() {
    size_t buf_size = w * h;
    CUDA_CHECK_RETURN(
        cudaMalloc(&curr_reservoirs, buf_size * sizeof(ReSTIRReservoir)));
    CUDA_CHECK_RETURN(
        cudaMemset(curr_reservoirs, 0, buf_size * sizeof(ReSTIRReservoir)));
    CUDA_CHECK_RETURN(
        cudaMalloc(&prev_reservoirs, buf_size * sizeof(ReSTIRReservoir)));
    CUDA_CHECK_RETURN(
        cudaMemset(prev_reservoirs, 0, buf_size * sizeof(ReSTIRReservoir)));
    CUDA_CHECK_RETURN(cudaMalloc(&g_buffer, buf_size * sizeof(ReSTIRGBuffer)));
    CUDA_CHECK_RETURN(
        cudaMemset(g_buffer, 0, buf_size * sizeof(ReSTIRGBuffer)));
}

void ReSTIRPathTracer::swap_reservoirs() {
    ReSTIRReservoir *tmp = prev_reservoirs;
    prev_reservoirs = curr_reservoirs;
    curr_reservoirs = tmp;
    // Clear current reservoirs for next frame
    CUDA_CHECK_RETURN(
        cudaMemset(curr_reservoirs, 0, w * h * sizeof(ReSTIRReservoir)));
}

ReSTIRPathTracer::ReSTIRPathTracer(const Scene &scene, bool temporal)
    : PathTracer(scene, false), cam_vol_id(scene.cam_vol_id),
      enable_temporal(temporal), curr_reservoirs(nullptr),
      prev_reservoirs(nullptr), g_buffer(nullptr) {
    media = scene.media;
    allocate_restir_buffers();
}

ReSTIRPathTracer::~ReSTIRPathTracer() {
    CUDA_CHECK_RETURN(cudaFree(curr_reservoirs));
    CUDA_CHECK_RETURN(cudaFree(prev_reservoirs));
    CUDA_CHECK_RETURN(cudaFree(g_buffer));
    printf("[Renderer] ReSTIR Path Tracer Object destroyed.\n");
}

CPT_CPU std::vector<uint8_t> ReSTIRPathTracer::render(const MaxDepthParams &md,
                                                      int num_iter,
                                                      bool gamma_correction) {
    printf("ReSTIR Rendering starts (M=%d, temporal=%s).\n", md.restir_m,
           enable_temporal ? "on" : "off");
    TicToc _timer("render_pt_restir_kernel()", num_iter);
    size_t cached_size = std::max(num_cache * sizeof(uint4), sizeof(uint4));
    dim3 grid(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y);
    dim3 block(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y);
    dim3 sp_grid((w + 15) >> 4, (h + 15) >> 4);
    dim3 sp_block(16, 16);

    for (int i = 0; i < num_iter; i++) {
        render_pt_restir_kernel<false><<<grid, block, cached_size>>>(
            *camera, verts, norms, uvs, media, obj_info, emitter_prims,
            bvh_leaves, nodes, _cached_nodes, image, md, output_buffer, nullptr,
            num_emitter, i * SEED_SCALER + seed_offset, cam_vol_id, num_nodes,
            accum_cnt, num_cache, envmap_id, false, nullptr, curr_reservoirs,
            g_buffer, false);

        restir_spatial_reuse_kernel<<<sp_grid, sp_block>>>(
            media, curr_reservoirs, g_buffer, w, h, envmap_id, SPATIAL_PASSES,
            i * SEED_SCALER + seed_offset + 1);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printProgress(i, num_iter);
    }
    printf("\n");
    return image.export_cpu(1.f / num_iter, gamma_correction);
}

CPT_CPU void ReSTIRPathTracer::render_online(const MaxDepthParams &md,
                                             bool gamma_corr) {
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0,
           cached_size = std::max(num_cache * sizeof(uint4), sizeof(uint4));
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer(
        (void **)&output_buffer, &_num_bytes, pbo_resc));
    accum_cnt++;

    dim3 grid(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y);
    dim3 block(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y);
    dim3 sp_grid((w + 15) >> 4, (h + 15) >> 4);
    dim3 sp_block(16, 16);

    render_pt_restir_kernel<true><<<grid, block, cached_size>>>(
        *camera, verts, norms, uvs, media, obj_info, emitter_prims, bvh_leaves,
        nodes, _cached_nodes, image, md, output_buffer, nullptr, num_emitter,
        accum_cnt * SEED_SCALER + seed_offset, cam_vol_id, num_nodes, accum_cnt,
        num_cache, envmap_id, gamma_corr, prev_reservoirs, curr_reservoirs,
        g_buffer, enable_temporal);

    restir_spatial_reuse_kernel<<<sp_grid, sp_block>>>(
        media, curr_reservoirs, g_buffer, w, h, envmap_id, SPATIAL_PASSES,
        accum_cnt * SEED_SCALER + seed_offset + 1);

    if (enable_temporal) {
        swap_reservoirs();
    }
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}

CPT_CPU const float *ReSTIRPathTracer::render_raw(const MaxDepthParams &md,
                                                  bool gamma_corr) {
    size_t cached_size = std::max(num_cache * sizeof(uint4), sizeof(uint4));
    accum_cnt++;

    dim3 grid(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y);
    dim3 block(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y);
    dim3 sp_grid((w + 15) >> 4, (h + 15) >> 4);
    dim3 sp_block(16, 16);

    render_pt_restir_kernel<true><<<grid, block, cached_size>>>(
        *camera, verts, norms, uvs, media, obj_info, emitter_prims, bvh_leaves,
        nodes, _cached_nodes, image, md, output_buffer, var_buffer, num_emitter,
        accum_cnt * SEED_SCALER + seed_offset, cam_vol_id, num_nodes, accum_cnt,
        num_cache, envmap_id, gamma_corr, prev_reservoirs, curr_reservoirs,
        g_buffer, enable_temporal);

    restir_spatial_reuse_kernel<<<sp_grid, sp_block>>>(
        media, curr_reservoirs, g_buffer, w, h, envmap_id, SPATIAL_PASSES,
        accum_cnt * SEED_SCALER + seed_offset + 1);

    if (enable_temporal) {
        swap_reservoirs();
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return output_buffer;
}
