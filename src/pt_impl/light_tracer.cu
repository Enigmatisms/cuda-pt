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
 * @brief Megakernel Path Tracer implementation
 * @date 2024.10.10
 */

#include "renderer/light_tracer.cuh"

static constexpr int SEED_SCALER = 11467; //-4!
static constexpr int SHFL_THREAD_X =
    4; // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16
       // threads
static constexpr int SHFL_THREAD_Y =
    3; // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16
       // threads

CPT_CPU std::vector<uint8_t> LightTracer::render(const MaxDepthParams &md,
                                                 int num_iter,
                                                 bool gamma_correction) {
    printf("Rendering starts.\n");
    TicToc _timer("render_lt_kernel()", num_iter);
    size_t cached_size = num_cache * sizeof(uint4);
    for (int i = 0; i < num_iter; i++) {
        // for more sophisticated renderer (like path tracer), shared_memory
        // should be used
        if (bidirectional) {
            render_pt_kernel<SingleTileScheduler, true>
                <<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
                   dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
                    *camera, verts, norms, uvs, obj_info, emitter_prims,
                    bvh_leaves, nodes, _cached_nodes, image, md, nullptr,
                    nullptr, num_emitter, accum_cnt * SEED_SCALER + seed_offset,
                    num_nodes, accum_cnt);
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }
        render_lt_kernel<false>
            <<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
               dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
                *camera, verts, norms, uvs, obj_info, emitter_prims, bvh_leaves,
                nodes, _cached_nodes, image, md, nullptr, num_emitter,
                i * SEED_SCALER + seed_offset, num_nodes, spec_constraint);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printProgress(i, num_iter);
    }
    printf("\n");
    return image.export_cpu(1.f / num_iter, gamma_correction, true);
}

CPT_CPU void LightTracer::render_online(const MaxDepthParams &md,
                                        bool gamma_corr) {
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0, cached_size = num_cache * sizeof(uint4);
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer(
        (void **)&output_buffer, &_num_bytes, pbo_resc));

    accum_cnt++;
    if (bidirectional) {
        render_pt_kernel<SingleTileScheduler, false>
            <<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
               dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
                *camera, verts, norms, uvs, obj_info, emitter_prims, bvh_leaves,
                nodes, _cached_nodes, image, md, output_buffer, nullptr,
                num_emitter, accum_cnt * SEED_SCALER + seed_offset, num_nodes,
                accum_cnt, num_cache, false);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    render_lt_kernel<true>
        <<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
           dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
            *camera, verts, norms, uvs, obj_info, emitter_prims, bvh_leaves,
            nodes, _cached_nodes, image, md, output_buffer, num_emitter,
            accum_cnt * SEED_SCALER + seed_offset, num_nodes, accum_cnt,
            num_cache, spec_constraint, caustic_scaling, gamma_corr);
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}

CPT_CPU const float *LightTracer::render_raw(const MaxDepthParams &md,
                                             bool gamma_corr) {
    size_t cached_size = std::max(num_cache * sizeof(uint4), sizeof(uint4));
    accum_cnt++;
    if (bidirectional) {
        render_pt_kernel<SingleTileScheduler, false>
            <<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
               dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
                *camera, verts, norms, uvs, obj_info, emitter_prims, bvh_leaves,
                nodes, _cached_nodes, image, md, output_buffer, nullptr,
                num_emitter, accum_cnt * SEED_SCALER + seed_offset, num_nodes,
                accum_cnt, num_cache, false);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
    render_lt_kernel<true>
        <<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
           dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
            *camera, verts, norms, uvs, obj_info, emitter_prims, bvh_leaves,
            nodes, _cached_nodes, image, md, output_buffer, num_emitter,
            accum_cnt * SEED_SCALER + seed_offset, num_nodes, accum_cnt,
            num_cache, spec_constraint, caustic_scaling, gamma_corr);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return output_buffer;
}
