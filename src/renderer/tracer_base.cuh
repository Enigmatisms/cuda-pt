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
 * @brief Base class of path tracers
 * @date: 2024.5.12
 */
#pragma once
#include "core/host_device.cuh"
#include "core/max_depth.h"
#include "core/primitives.cuh"
#include "core/scene.cuh"

class TracerBase {
  protected:
    PrecomputedArray verts;
    NormalArray norms;
    ConstBuffer<PackedHalf2> uvs;
    DeviceImage image;
    int num_prims;
    int w;
    int h;
    int seed_offset;

    DeviceCamera *camera;
    float *output_buffer; // output buffer for images
    float *var_buffer;    // variance buffer

    // IMGUI related
    int accum_cnt;
    uint32_t cuda_texture_id, pbo_id;
    cudaGraphicsResource_t pbo_resc;

  public:
    std::vector<char> serialized_data;

  public:
    /**
     * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3,
     * 3D)
     * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3,
     * 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
     */
    TracerBase(const Scene &scene)
        : verts(scene.num_prims), norms(scene.num_prims), uvs(scene.num_prims),
          image(scene.config.width, scene.config.height),
          num_prims(scene.num_prims), w(scene.config.width),
          h(scene.config.height), seed_offset(0), output_buffer(nullptr),
          var_buffer(nullptr), accum_cnt(0), cuda_texture_id(0), pbo_id(0) {
        scene.export_prims(verts, norms, uvs);
    }

    virtual ~TracerBase() {
        image.destroy();
        verts.destroy();
        norms.destroy();
        uvs.destroy();
    }

    CPT_CPU uint32_t &get_texture_id() noexcept {
        return this->cuda_texture_id;
    }
    CPT_CPU uint32_t &get_pbo_id() noexcept { return this->pbo_id; }

    // set parameters with serialized data
    virtual CPT_CPU void param_setter(const std::vector<char> &bytes) {}

    virtual CPT_CPU std::vector<uint8_t> render(const MaxDepthParams &md,
                                                int num_iter = 64,
                                                bool gamma_correction = true) {
        throw std::runtime_error("Not implemented.\n");
        return {};
    }

    virtual CPT_CPU void render_online(
        const MaxDepthParams &md,
        bool gamma_corr = false /* whether to enable gamma correction*/
    ) {
        throw std::runtime_error("Not implemented.\n");
    }

    // Render the scene once (1 spp) and output the output_buffer
    virtual CPT_CPU const float *render_raw(const MaxDepthParams &md,
                                            bool gamma_corr = false) {
        throw std::runtime_error("Not implemented.\n");
    }

    virtual CPT_CPU const float *get_variance_buffer() const {
        return var_buffer;
    }

    /**
     * @brief initialize graphics resources
     * @param executor the callback function pointer
     */
    CPT_CPU void graphics_resc_init(void (*executor)(cudaGraphicsResource_t &,
                                                     uint32_t &, uint32_t &,
                                                     int, int)) {
        executor(pbo_resc, pbo_id, cuda_texture_id, w, h);
        initialize_output_buffer();
    }

    CPT_CPU void initialize_output_buffer() {
        // Allocate accumulation buffer
        CUDA_CHECK_RETURN(
            cudaMalloc(&output_buffer, w * h * 4 * sizeof(float)));
        CUDA_CHECK_RETURN(
            cudaMemset(output_buffer, 0, w * h * 4 * sizeof(float)));
    }

    CPT_CPU void initialize_var_buffer() {
        // Allocate variance buffer
        CUDA_CHECK_RETURN(cudaMalloc(&var_buffer, w * h * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemset(var_buffer, 0, w * h * sizeof(float)));
    }

    CPT_CPU_INLINE void reset_out_buffer() {
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        CUDA_CHECK_RETURN(
            cudaMemset(image.data(), 0,
                       w * h * sizeof(float4))); // reset image buffer
        accum_cnt = 0;                           // reset accumulation counter
    }

    CPT_CPU void update_camera(const DeviceCamera *const cam) {
        CUDA_CHECK_RETURN(cudaMemcpyAsync(camera, cam, sizeof(DeviceCamera),
                                          cudaMemcpyHostToDevice));
    }

    CPT_CPU int get_num_sample() const noexcept { return accum_cnt; }

    virtual CPT_CPU std::vector<uint8_t>
    get_image_buffer(bool gamma_cor) const {
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        return image.export_cpu(1.f / accum_cnt, gamma_cor);
    }

    CPT_CPU int width() const noexcept { return this->w; }
    CPT_CPU int height() const noexcept { return this->h; }
    CPT_CPU int cnt() const noexcept { return this->accum_cnt; }
    CPT_CPU void set_seed_offset(int val) { this->seed_offset = val; }
};
