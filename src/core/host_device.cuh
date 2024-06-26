/**
 * Host Device memory transaction util function
 * @date: 5.5.2024
 * @author: Qianyue He 
*/
#pragma once
#include <vector>
#include "core/cuda_utils.cuh"
#include "core/vec4.cuh"
#include <vector_types.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename Ty>
std::decay_t<Ty>* to_gpu(Ty&& object) {
    using BaseTy = std::decay_t<Ty>;
    BaseTy* ptr;
    CUDA_CHECK_RETURN(cudaMalloc(&ptr, sizeof(BaseTy)));
    CUDA_CHECK_RETURN(cudaMemcpy(ptr, &object, sizeof(BaseTy), cudaMemcpyHostToDevice));
    return ptr;
}

template <typename Ty>
__global__ void parallel_memset(Ty* dst, Ty value, int length) {
    int num_thread = blockDim.x;
    for (int i = 0; i < length; i += num_thread) {
        dst[threadIdx.x + num_thread] = value;
    }
}

template <typename Ty>
Ty* allocate_copy_managed(const std::vector<Ty>& mem_src) {
    Ty* dev_mem = nullptr;
    size_t bytes = mem_src.size() * sizeof(Ty);
    CUDA_CHECK_RETURN(cudaMallocManaged(&dev_mem, bytes));
    CUDA_CHECK_RETURN(cudaMemcpy(dev_mem, mem_src.data(), bytes, cudaMemcpyHostToDevice));
    return dev_mem;
}

template <typename Ty>
Ty* make_filled_memory(Ty fill_value, size_t length) {
    Ty* dev_mem = nullptr;
    size_t bytes = length * sizeof(Ty);
    CUDA_CHECK_RETURN(cudaMallocManaged(&dev_mem, bytes));
    parallel_memset<Ty><<<1, 256>>>(dev_mem, fill_value, length);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return dev_mem;
}

class DeviceImage {
private:
    Vec4* image_buffer;
    Vec4* host_buffer;
    size_t pitch;
    int _w;
    int _h;
public:
    DeviceImage(int width = 800, int height = 800): pitch(0), _w(width), _h(height) {
        host_buffer = new Vec4[_w * _h];
        CUDA_CHECK_RETURN(cudaMallocPitch(&image_buffer, &pitch, width * sizeof(Vec4), height));
        CUDA_CHECK_RETURN(cudaMemset2D(image_buffer, pitch, 0, width * sizeof(Vec4), height));
    }

    ~DeviceImage() {
        delete [] host_buffer;
        cudaFree(image_buffer);
    }

    // TODO: can be accelerated via multi-threading
    std::vector<uint8_t> export_cpu(float inv_factor = 1, bool gamma_cor = true) const {
        std::vector<uint8_t> byte_buffer(_w * _h * 4);
        size_t copy_pitch = _w * sizeof(Vec4);
        CUDA_CHECK_RETURN(cudaMemcpy2D(host_buffer, copy_pitch, image_buffer, pitch, copy_pitch, _h, cudaMemcpyDeviceToHost));
        if (gamma_cor) {
            for (int i = 0; i < _h; i ++) {
                int base = i * _w;
                for (int j = 0; j < _w; j ++) {
                    int pixel_index = base + j;
                    const Vec4& color = host_buffer[pixel_index];
                    pixel_index <<= 2;
                    byte_buffer[pixel_index]     = to_int(color.x() * inv_factor);
                    byte_buffer[pixel_index + 1] = to_int(color.y() * inv_factor);
                    byte_buffer[pixel_index + 2] = to_int(color.z() * inv_factor);
                    byte_buffer[pixel_index + 3] = 255;
                }
            }
        } else {
            for (int i = 0; i < _h; i ++) {
                int base = i * _w;
                for (int j = 0; j < _w; j ++) {
                    int pixel_index = base + j;
                    const Vec4& color = host_buffer[pixel_index];
                    pixel_index <<= 2;
                    byte_buffer[pixel_index]     = to_int_linear(color.x() * inv_factor);
                    byte_buffer[pixel_index + 1] = to_int_linear(color.y() * inv_factor);
                    byte_buffer[pixel_index + 2] = to_int_linear(color.z() * inv_factor);
                    byte_buffer[pixel_index + 3] = 255;
                }
            }
        }
        return byte_buffer;
    }

    // for cudaMallocPitch (with extra memory alignment), we need to use pitch to access
    CPT_CPU_GPU Vec4& operator() (int col, int row) {  return ((Vec4*)((char*)image_buffer + row * pitch))[col]; }
    CPT_CPU_GPU const Vec4& operator() (int col, int row) const { return ((Vec4*)((char*)image_buffer + row * pitch))[col]; }

    CPT_CPU_GPU int w() const noexcept { return _w; }
    CPT_CPU_GPU int h() const noexcept { return _h; }
};
