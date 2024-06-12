/**
*/
#pragma once
#include "core/vec2.cuh"
#include "core/host_device.cuh"

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
            data[3 * i]     = vec1[i];
            data[3 * i + 1] = vec2[i];
            data[3 * i + 2] = vec3[i];
        }
    }

    CPT_CPU void fill(const StructType& v) {
        parallel_memset<<<1, 256>>>(data, v, size * 3);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
};

template <typename StructType>
using ConstAoS3Ptr = const AoS3<StructType>* const;
using ConstPrimAoSPtr = const AoS3<Vec3>* const;
using ConstUVAoSPtr   = const AoS3<Vec2>* const;