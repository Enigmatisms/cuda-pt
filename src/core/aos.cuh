/**
*/
#pragma once
#include "core/vec2.cuh"
#include "core/host_device.cuh"
#define TRI_IDX(index) (index << 1) + index
template <typename StructType>
class AoS3 {
public:
    StructType* data;
    size_t size;
public:
    CPT_CPU AoS3(size_t _size): size(_size) {
        CUDA_CHECK_RETURN(cudaMallocManaged(&data, 3 * sizeof(StructType) * _size));
    }

    CPT_CPU void destroy() { 
        size = 0;
        CUDA_CHECK_RETURN(cudaFree(data)); 
    }

    // GPU implements SoA constructor very differently
    CPT_GPU AoS3(StructType* _data, size_t size): 
        data(_data), size(size) {}

    CPT_CPU void from_vectors(
        const std::vector<StructType>& vec1,
        const std::vector<StructType>& vec2,
        const std::vector<StructType>& vec3
    ) {
        for (size_t i = 0; i < size; i ++) {
            data[TRI_IDX(i)]     = vec1[i];
            data[TRI_IDX(i) + 1] = vec2[i];
            data[TRI_IDX(i) + 2] = vec3[i];
        }
    }

    CPT_CPU_GPU const StructType& x(int index) const { return data[TRI_IDX(index)]; }
    CPT_CPU_GPU const StructType& y(int index) const { return data[TRI_IDX(index) + 1]; }
    CPT_CPU_GPU const StructType& z(int index) const { return data[TRI_IDX(index) + 2]; }

    CPT_CPU_GPU StructType& x(int index) { return data[TRI_IDX(index)]; }
    CPT_CPU_GPU StructType& y(int index) { return data[TRI_IDX(index) + 1]; }
    CPT_CPU_GPU StructType& z(int index) { return data[TRI_IDX(index) + 2]; }

    CPT_CPU void fill(const StructType& v) {
        parallel_memset<<<1, 256>>>(data, v, size * 3);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
};

template <typename StructType>
using ConstAoS3Ptr = const AoS3<StructType>* const;
using ConstPrimPtr = const AoS3<Vec3>* const;
using ConstUVPtr   = const AoS3<Vec2>* const;