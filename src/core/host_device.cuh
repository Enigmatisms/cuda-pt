/**
 * Host Device memory transaction util function
 * @date: 5.5.2024
 * @author: Qianyue He 
*/
#pragma once
#include <vector>
#include "cuda_utils.cuh"
#include <vector_types.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

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
    parallel_memset<<<1, 256>>><Ty>(dev_mem, fill_value, length);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return dev_mem;
}

class DeviceImage {
private:
    Vec3* image_buffer;
    Vec3* host_buffer;
    size_t pitch;
    int _w;
    int _h;
public:
    DeviceImage(int width = 800, int height = 800): pitch(0), _w(width), _h(height) {
        host_buffer = new Vec3[_w * _h];
        CUDA_CHECK_RETURN(cudaMallocPitch(&image_buffer, &pitch, width * sizeof(Vec3), height));
        CUDA_CHECK_RETURN(cudaMemset2D(image_buffer, pitch, 0, width * sizeof(Vec3), height));
    }

    ~DeviceImage() {
        delete [] host_buffer;
        cudaFree(image_buffer);
    }

    // TODO: can be accelerated via multi-threading
    std::vector<uint8_t> export_cpu() const {
        std::vector<uint8_t> byte_buffer(_w * _h * 4);
        size_t copy_pitch = _w * sizeof(Vec3);
        CUDA_CHECK_RETURN(cudaMemcpy2D(host_buffer, copy_pitch, image_buffer, pitch, copy_pitch, _h, cudaMemcpyDeviceToHost));
        for (int i = 0; i < _h; i ++) {
            for (int j = 0; j < _w; j ++) {
                int pixel_index = i * _w + j;
                const Vec3& color = host_buffer[pixel_index];
                byte_buffer[(pixel_index << 2)]     = to_int(color.x());
                byte_buffer[(pixel_index << 2) + 1] = to_int(color.y());
                byte_buffer[(pixel_index << 2) + 2] = to_int(color.z());
                byte_buffer[(pixel_index << 2) + 3] = 255;
            }
        }
        return byte_buffer;
    }

    CPT_CPU_GPU Vec3& operator() (int x, int y) { return image_buffer[y * _w + x]; }
    CPT_CPU_GPU const Vec3& operator() (int x, int y) const { return image_buffer[y * _w + x]; }

    CPT_CPU_GPU int w() const noexcept { return _w; }
    CPT_CPU_GPU int h() const noexcept { return _h; }
};
