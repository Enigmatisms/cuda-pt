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
 * @date: 2024.5.5
 * @brief Host Device memory transaction util function
 * @author: Qianyue He
 */
#pragma once
#include "core/vec4.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <vector_types.h>

class DeviceImage {
  private:
    Vec4 *host_buffer;
    Vec4 *_buffer;
    int _w;
    int _h;

  public:
    DeviceImage(int width = 800, int height = 800) : _w(width), _h(height) {
        host_buffer = new Vec4[width * height];
        CUDA_CHECK_RETURN(cudaMalloc(&_buffer, width * height * sizeof(Vec4)));
        CUDA_CHECK_RETURN(
            cudaMemset(_buffer, 0, width * height * sizeof(Vec4)));
    }

    // manually call the destroy to deallocate memory
    void destroy() {
        CUDA_CHECK_RETURN(cudaFree(_buffer));
        delete[] host_buffer;
    }

    CPT_CPU_GPU_INLINE Vec4 &operator()(int col, int row) {
        return _buffer[row * _w + col];
    }
    CPT_CPU_GPU_INLINE const Vec4 &operator()(int col, int row) const {
        return _buffer[row * _w + col];
    }

    CPT_CPU_GPU_INLINE int w() const noexcept { return _w; }
    CPT_CPU_GPU_INLINE int h() const noexcept { return _h; }

    CPT_CPU_GPU_INLINE Vec4 *data() { return _buffer; }

    std::vector<uint8_t> export_cpu(float inv_factor = 1, bool gamma_cor = true,
                                    bool alpha_avg = false) const {
        std::vector<uint8_t> byte_buffer(_w * _h * 4);
        size_t copy_size = _w * _h * sizeof(Vec4);
        CUDA_CHECK_RETURN(cudaMemcpy(host_buffer, _buffer, copy_size,
                                     cudaMemcpyDeviceToHost));
        if (gamma_cor) {
            for (int i = 0; i < _h; i++) {
                int base = i * _w;
                for (int j = 0; j < _w; j++) {
                    int pixel_index = base + j;
                    const Vec4 &color = host_buffer[pixel_index];
                    pixel_index <<= 2;
                    byte_buffer[pixel_index + 3] = 255;
                    if (alpha_avg) {
                        if (color.w() < 1e-5f)
                            continue;
                        inv_factor = 1.f / color.w();
                    }
                    byte_buffer[pixel_index] = to_int(color.x() * inv_factor);
                    byte_buffer[pixel_index + 1] =
                        to_int(color.y() * inv_factor);
                    byte_buffer[pixel_index + 2] =
                        to_int(color.z() * inv_factor);
                }
            }
        } else {
            for (int i = 0; i < _h; i++) {
                int base = i * _w;
                for (int j = 0; j < _w; j++) {
                    int pixel_index = base + j;
                    const Vec4 &color = host_buffer[pixel_index];
                    pixel_index <<= 2;
                    byte_buffer[pixel_index + 3] = 255;
                    if (alpha_avg) {
                        if (color.w() < 1e-5f)
                            continue;
                        inv_factor = 1.f / color.w();
                    }
                    byte_buffer[pixel_index] =
                        to_int_linear(color.x() * inv_factor);
                    byte_buffer[pixel_index + 1] =
                        to_int_linear(color.y() * inv_factor);
                    byte_buffer[pixel_index + 2] =
                        to_int_linear(color.z() * inv_factor);
                }
            }
        }
        return byte_buffer;
    }
};
