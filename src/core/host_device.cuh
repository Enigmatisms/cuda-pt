/**
 * Host Device memory transaction util function
 * @date: 5.5.2024
 * @author: Qianyue He 
*/
#pragma once
#include <vector>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core/vec4.cuh"

template <typename Ty>
CPT_KERNEL void parallel_memset(Ty* dst, Ty value, int length) {
    int num_thread = blockDim.x;
    for (int i = 0; i < length; i += num_thread) {
        dst[threadIdx.x + num_thread] = value;
    }
}

class DeviceImage {
private:
    Vec4* host_buffer;
    Vec4* _buffer;
    int _w;
    int _h;
public:
    DeviceImage(int width = 800, int height = 800): _w(width), _h(height) {
        host_buffer = new Vec4[width * height];
        CUDA_CHECK_RETURN(cudaMalloc(&_buffer, width * height * sizeof(Vec4)));
        CUDA_CHECK_RETURN(cudaMemset(_buffer, 0, width * height * sizeof(Vec4)));
    }

    // manually call the destroy to deallocate memory
    void destroy() {
        CUDA_CHECK_RETURN(cudaFree(_buffer));
        delete [] host_buffer;
    }

    CPT_CPU_GPU_INLINE Vec4& operator() (int col, int row) {  return _buffer[row * _w + col]; }
    CPT_CPU_GPU_INLINE const Vec4& operator() (int col, int row) const { return _buffer[row * _w + col]; }

    CPT_CPU_GPU_INLINE int w() const noexcept { return _w; }
    CPT_CPU_GPU_INLINE int h() const noexcept { return _h; }

    CPT_CPU_GPU_INLINE Vec4* data() {
        return _buffer;
    }

    std::vector<uint8_t> export_cpu(float inv_factor = 1, bool gamma_cor = true, bool alpha_avg = false) const {
        std::vector<uint8_t> byte_buffer(_w * _h * 4);
        size_t copy_size = _w * _h * sizeof(Vec4);
        CUDA_CHECK_RETURN(cudaMemcpy(host_buffer, _buffer, copy_size, cudaMemcpyDeviceToHost));
        if (gamma_cor) {
            for (int i = 0; i < _h; i ++) {
                int base = i * _w;
                for (int j = 0; j < _w; j ++) {
                    int pixel_index = base + j;
                    const Vec4& color = host_buffer[pixel_index];
                    pixel_index <<= 2;
                    byte_buffer[pixel_index + 3] = 255;
                    if (alpha_avg) {
                        if (color.w() < 1e-5f) continue;
                        inv_factor = 1.f / color.w();
                    }
                    byte_buffer[pixel_index]     = to_int(color.x() * inv_factor);
                    byte_buffer[pixel_index + 1] = to_int(color.y() * inv_factor);
                    byte_buffer[pixel_index + 2] = to_int(color.z() * inv_factor);
                }
            }
        } else {
            for (int i = 0; i < _h; i ++) {
                int base = i * _w;
                for (int j = 0; j < _w; j ++) {
                    int pixel_index = base + j;
                    const Vec4& color = host_buffer[pixel_index];
                    pixel_index <<= 2;
                    byte_buffer[pixel_index + 3] = 255;
                    if (alpha_avg) {
                        if (color.w() < 1e-5f) continue;
                        inv_factor = 1.f / color.w();
                    }
                    byte_buffer[pixel_index]     = to_int_linear(color.x() * inv_factor);
                    byte_buffer[pixel_index + 1] = to_int_linear(color.y() * inv_factor);
                    byte_buffer[pixel_index + 2] = to_int_linear(color.z() * inv_factor);
                }
            }
        }
        return byte_buffer;
    }
};
