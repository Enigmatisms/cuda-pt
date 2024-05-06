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

using Pixel3 = float3;

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
    Pixel3* image_buffer;
    int _w;
    int _h;
public:
    DeviceImage(int width = 800, int height = 800): _w(width), _h(height) {
        cudaExtent extent = make_cudaExtent(width * sizeof(Pixel3), height, 1);
        Pixel3* image_buffer = nullptr;
        cudaPitchedPtr d_pitchedPtr;
        CUDA_CHECK_RETURN(cudaMalloc3D(&d_pitchedPtr, extent));
        image_buffer = (Pixel3*)d_pitchedPtr.ptr;
    }

    ~DeviceImage() {
        cudaFree(image_buffer);
    }

    void export_cpu() {
        throw std::runtime_error("Not implemented yet.");
    }

    CPT_CPU_GPU Pixel3& operator() (int x, int y) { return image_buffer[y * _w + x]; }
    CPT_CPU_GPU const Pixel3& operator() (int x, int y) const { return image_buffer[y * _w + x]; }

    CPT_CPU_GPU int w() const noexcept { return _w; }
    CPT_CPU_GPU int h() const noexcept { return _h; }
};
