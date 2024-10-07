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
std::decay_t<Ty>* to_gpu(Ty&& object) {
    using BaseTy = std::decay_t<Ty>;
    BaseTy* ptr;
    CUDA_CHECK_RETURN(cudaMalloc(&ptr, sizeof(BaseTy)));
    CUDA_CHECK_RETURN(cudaMemcpy(ptr, &object, sizeof(BaseTy), cudaMemcpyHostToDevice));
    return ptr;
}

template <typename Ty>
CPT_KERNEL void parallel_memset(Ty* dst, Ty value, int length) {
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

/**
 * Accessor for CUDA pitch array
*/
template<typename ValType>
struct Accessor {
    CPT_CPU_GPU Accessor() {}
    CPT_CPU_GPU_INLINE ValType& operator() (ValType* buffer, uint32_t col, uint32_t row, uint32_t pitch) 
                {  return ((ValType*)((char*)buffer + row * pitch))[col]; }
    CPT_CPU_GPU_INLINE const ValType& operator() (ValType* buffer, uint32_t col, uint32_t row, uint32_t pitch) const 
                { return ((ValType*)((char*)buffer + row * pitch))[col]; }
};

/**
 * A 2D CUDA pitch array encapsulation
*/
template<typename ValType>
class DevicePitchBuffer {
protected:
    ValType* _buffer;
    size_t pitch;
    int _w;
    int _h;
public:
    DevicePitchBuffer(int width = 800, int height = 800): pitch(0), _w(width), _h(height) {
        CUDA_CHECK_RETURN(cudaMallocPitch(&_buffer, &pitch, width * sizeof(ValType), height));
        CUDA_CHECK_RETURN(cudaMemset2D(_buffer, pitch, 0, width * sizeof(ValType), height));
    }

    ~DevicePitchBuffer() {
        cudaFree(_buffer);
    }

    // for cudaMallocPitch (with extra memory alignment), we need to use pitch to access
    CPT_CPU_GPU_INLINE ValType& operator() (int col, int row) {  return ((ValType*)((char*)_buffer + row * pitch))[col]; }
    CPT_CPU_GPU_INLINE const ValType& operator() (int col, int row) const { return ((ValType*)((char*)_buffer + row * pitch))[col]; }

    CPT_CPU_GPU_INLINE int w() const noexcept { return _w; }
    CPT_CPU_GPU_INLINE int h() const noexcept { return _h; }
    CPT_CPU_GPU_INLINE int get_pitch() const noexcept { return pitch; }

    CPT_CPU_GPU_INLINE static Accessor<ValType> accessor() noexcept {
        return Accessor<ValType>();
    }

    CPT_CPU_GPU_INLINE ValType* data() {
        return _buffer;
    }
};

class DeviceImage: public DevicePitchBuffer<Vec4> {
private:
    Vec4* host_buffer;
public:
    DeviceImage(int width = 800, int height = 800): DevicePitchBuffer<Vec4>(width, height) {
        host_buffer = new Vec4[width * height];
    }

    ~DeviceImage() {
        delete [] host_buffer;
    }

    std::vector<uint8_t> export_cpu(float inv_factor = 1, bool gamma_cor = true, bool alpha_avg = false) const {
        std::vector<uint8_t> byte_buffer(_w * _h * 4);
        size_t copy_pitch = _w * sizeof(Vec4);
        CUDA_CHECK_RETURN(cudaMemcpy2D(host_buffer, copy_pitch, _buffer, pitch, copy_pitch, _h, cudaMemcpyDeviceToHost));
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
