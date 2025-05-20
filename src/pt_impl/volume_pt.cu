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
 * @brief Megakernel Volumetric Path Tracer implementation
 * @date 2025.2.9
 */
#include "renderer/megakernel_vpt.cuh"
#include "renderer/volume_pt.cuh"

static constexpr int SEED_SCALER =
    11467; // 11451 is not a prime, while 11467 is
static constexpr int SHFL_THREAD_X =
    5; // blockDim.x: 1 << SHFL_THREAD_X, by default, SHFL_THREAD_X is 4: 16
       // threads
static constexpr int SHFL_THREAD_Y =
    2; // blockDim.y: 1 << SHFL_THREAD_Y, by default, SHFL_THREAD_Y is 4: 16
       // threads

VolumePathTracer::VolumePathTracer(const Scene &scene)
    : PathTracer(scene, false), cam_vol_id(scene.cam_vol_id) {
    media = scene.media;
}

VolumePathTracer::~VolumePathTracer() {
    printf("[Renderer] Volume Path Tracer Object destroyed.\n");
}

CPT_CPU std::vector<uint8_t> VolumePathTracer::render(const MaxDepthParams &md,
                                                      int num_iter,
                                                      bool gamma_correction) {
    printf("Rendering starts.\n");
    TicToc _timer("render_pt_kernel()", num_iter);
    size_t cached_size = std::max(num_cache * sizeof(uint4), sizeof(uint4));
    for (int i = 0; i < num_iter; i++) {
        // for more sophisticated renderer (like path tracer), shared_memory
        // should be used
        render_vpt_kernel<false>
            <<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
               dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
                *camera, verts, norms, uvs, media, obj_info, emitter_prims,
                bvh_leaves, nodes, _cached_nodes, image, md, output_buffer,
                nullptr, num_emitter, i * SEED_SCALER + seed_offset, cam_vol_id,
                num_nodes, accum_cnt, num_cache, envmap_id);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        printProgress(i, num_iter);
    }
    printf("\n");
    return image.export_cpu(1.f / num_iter, gamma_correction);
}

CPT_CPU void VolumePathTracer::render_online(const MaxDepthParams &md,
                                             bool gamma_corr) {
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
    size_t _num_bytes = 0,
           cached_size = std::max(num_cache * sizeof(uint4), sizeof(uint4));
    // if we have an illegal memory access here: check whether you have a valid
    // emitter in the xml scene description file. it might be possible that
    // having no valid emitter triggers an illegal memory access
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer(
        (void **)&output_buffer, &_num_bytes, pbo_resc));
    accum_cnt++;

    render_vpt_kernel<true>
        <<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
           dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
            *camera, verts, norms, uvs, media, obj_info, emitter_prims,
            bvh_leaves, nodes, _cached_nodes, image, md, output_buffer, nullptr,
            num_emitter, accum_cnt * SEED_SCALER + seed_offset, cam_vol_id,
            num_nodes, accum_cnt, num_cache, envmap_id, gamma_corr);
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
}

CPT_CPU const float *VolumePathTracer::render_raw(const MaxDepthParams &md,
                                                  bool gamma_corr) {
    size_t cached_size = std::max(num_cache * sizeof(uint4), sizeof(uint4));
    accum_cnt++;
    render_vpt_kernel<true>
        <<<dim3(w >> SHFL_THREAD_X, h >> SHFL_THREAD_Y),
           dim3(1 << SHFL_THREAD_X, 1 << SHFL_THREAD_Y), cached_size>>>(
            *camera, verts, norms, uvs, media, obj_info, emitter_prims,
            bvh_leaves, nodes, _cached_nodes, image, md, output_buffer,
            var_buffer, num_emitter, accum_cnt * SEED_SCALER + seed_offset,
            cam_vol_id, num_nodes, accum_cnt, num_cache, envmap_id, gamma_corr);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return output_buffer;
}
