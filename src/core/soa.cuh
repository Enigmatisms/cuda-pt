/**
 * Easy SoA (Struct of Arrays) encapsulation
 * SoA can facilitate coalesced memory access
 * and cache coherence
 * 
 * SoA3 aims to hold three different field in a array
 * for example, vec3 (x, y, z)
*/
#pragma once
#include "core/vec2.cuh"
#include "core/host_device.cuh"

template <typename StructType>
class SoA3 {
private:
    StructType* _data;
public:
    StructType* x;
    StructType* y;
    StructType* z;
    size_t size;
public:
    CPT_CPU SoA3(size_t _size): size(_size) {
        CUDA_CHECK_RETURN(cudaMallocManaged(&_data, 3 * sizeof(StructType) * _size));
        x = &_data[0];
        y = &_data[_size];
        z = &_data[_size << 1];
    }

    CPT_CPU ~SoA3() {
        CUDA_CHECK_RETURN(cudaFree(_data));
    }

    CPT_CPU void from_vectors(
        const std::vector<StructType>& vec1,
        const std::vector<StructType>& vec2,
        const std::vector<StructType>& vec3
    ) {
        CUDA_CHECK_RETURN(cudaMemcpy(x, vec1.data(), sizeof(StructType) * std::min(vec1.size(), size), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(y, vec2.data(), sizeof(StructType) * std::min(vec2.size(), size), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(z, vec3.data(), sizeof(StructType) * std::min(vec3.size(), size), cudaMemcpyHostToDevice));
    }

    CPT_CPU void fill(
        const StructType& v1,
        const StructType& v2,
        const StructType& v3
    ) {
        cudaStream_t stream[3];
        for (int i = 0; i < 3; i++) {
            cudaStreamCreate(&stream[i]);
            parallel_memset<<<1, 256, 0, stream[i]>>>(&_data[i * size], v1, size);
            cudaStreamDestroy(stream[i]);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }

    CPT_CPU void fill(const StructType& v) {
        parallel_memset<<<1, 256>>>(x, v, size * 3);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
};

template <typename StructType>
using ConstSoA3Ptr = const SoA3<StructType>* const;
using ConstPrimPtr = const SoA3<Vec3>* const;
using ConstUVPtr   = const SoA3<Vec2>* const;