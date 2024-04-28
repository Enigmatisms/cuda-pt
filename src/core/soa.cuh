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

    CPT_CPU ~SoA3():  {
        CUDA_CHECK_RETURN(cudaFree(_data));
    }
};

template <typename StructType>
using ConstSoA3Ptr = const SoA3<StructType>* const;
template <typename Ty>
using ConstPrimPtr = const SoA3<Vec3<Ty>>* const;
template <typename Ty>
using ConstUVPtr   = const SoA3<Vec2<Ty>>* const;