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
#include "core/vec4.cuh"
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
        const std::vector<StructType>& vec3,
        const std::vector<bool>* const sphere_flags = nullptr
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
class SoA3 {
public:
    StructType* data;
    size_t size;
public:
    CPT_CPU SoA3(size_t _size) : size(_size) {
        CUDA_CHECK_RETURN(cudaMallocManaged(&data, 3 * sizeof(StructType) * _size));
    }

    CPT_CPU void destroy() {
        size = 0;
        CUDA_CHECK_RETURN(cudaFree(data));
    }

    // GPU implements SoA constructor very differently
    CPT_GPU SoA3(StructType* _data, size_t size) :
        data(_data), size(size) {}

    CPT_CPU void from_vectors(
        const std::vector<StructType>& vec1,
        const std::vector<StructType>& vec2,
        const std::vector<StructType>& vec3,
        const std::vector<bool>* const sphere_flags = nullptr
    ) {
        for (size_t i = 0; i < size; i++) {
            data[i]               = vec1[i];
            data[size + i]        = vec2[i];
            data[(size << 1) + i] = vec3[i];
        }
    }

    CPT_CPU void fill(const StructType& v) {
        parallel_memset << <1, 256 >> > (data, v, size * 3);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }

    CPT_CPU_GPU const StructType& x(int index) const { return data[index]; }
    CPT_CPU_GPU const StructType& y(int index) const { return data[index + size]; }
    CPT_CPU_GPU const StructType& z(int index) const { return data[index + (size << 1)]; }

    CPT_CPU_GPU StructType& x(int index) { return data[index]; }
    CPT_CPU_GPU StructType& y(int index) { return data[index + size]; }
    CPT_CPU_GPU StructType& z(int index) { return data[index + (size << 1)]; }
};

// PrecomputeAoS stored triangle difference, two of the adjacent matrix terms
// and one Vec3 array for absolute positioning (of a triangle)
// ray intersection and emitter sampling efficiency can be boosted
// 9 floats per triangle -> 12 floats per triangle
class PrecomputeAoS {
public:
    Vec4* data;         // packed sphere data or vertex difference (with adjoint matrix terms packed)
    size_t size;
public:
    CPT_CPU PrecomputeAoS(size_t _size) : size(_size) {
        CUDA_CHECK_RETURN(cudaMallocManaged(&data, 3 * sizeof(Vec4) * _size));
    }

    CPT_CPU void destroy() {
        size = 0;
        CUDA_CHECK_RETURN(cudaFree(data));
    }

    // GPU implements SoA constructor very differently
    CPT_GPU PrecomputeAoS(Vec4* _data, size_t size) :
        data(_data), size(size) {}

    // vertices input, convert to float4 and store precomputed adjoint matrix value
    // convert from vertices to vertex difference
    CPT_CPU void from_vectors(
        const std::vector<Vec3>& vec1,
        const std::vector<Vec3>& vec2,
        const std::vector<Vec3>& vec3,
        const std::vector<bool>* const sphere_flags = nullptr         // use naked ptr to enable null
    ) {
        for (size_t i = 0; i < size; i++) {
            if (sphere_flags && sphere_flags->at(i)) {                 // sphere is compacted into 4 floats, fast loading!
                Vec3 center = vec1[i];
                data[TRI_IDX(i)]     = Vec4(center.x(), center.y(), center.z(), vec2[i].x());
                data[TRI_IDX(i) + 1] = Vec4();
                data[TRI_IDX(i) + 2] = Vec4();
                continue;
            }
            Vec3 abs1  = vec1[i],
                 diff1 = vec2[i] - abs1,
                 diff2 = vec3[i] - abs1;
            // precomputed adjacent matrix terms, work as padding (saved 3 FMADD, 3 FMUL and 3 FADD per triangle)
            float a20 = diff1.y() * diff2.z() - diff1.z() * diff2.y(),
                  a21 = diff2.x() * diff1.z() - diff1.x() * diff2.z(),
                  a22 = diff1.x() * diff2.y() - diff2.x() * diff1.y();
            data[TRI_IDX(i)]     = Vec4(abs1.x(), abs1.y(), abs1.z(), a20);
            data[TRI_IDX(i) + 1] = Vec4(diff1.x(), diff1.y(), diff1.z(), a21);
            data[TRI_IDX(i) + 2] = Vec4(diff2.x(), diff2.y(), diff2.z(), a22);
        }
    }

    CPT_CPU void fill(const Vec4& v) {
        parallel_memset << <1, 256 >> > (data, v, size * 3);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }

    CPT_CPU_GPU const Vec4& x(int index) const { return data[TRI_IDX(index)]; }
    CPT_CPU_GPU const Vec4& y(int index) const { return data[TRI_IDX(index) + 1]; }
    CPT_CPU_GPU const Vec4& z(int index) const { return data[TRI_IDX(index) + 2]; }

    CPT_CPU_GPU Vec4& x(int index) { return data[TRI_IDX(index)]; }
    CPT_CPU_GPU Vec4& y(int index) { return data[TRI_IDX(index) + 1]; }
    CPT_CPU_GPU Vec4& z(int index) { return data[TRI_IDX(index) + 2]; }


    CPT_CPU_GPU Vec3 x_clipped(int index) const { 
        auto v = data[TRI_IDX(index)]; 
        return Vec3(v.x(), v.y(), v.z());
    }
    CPT_CPU_GPU Vec3 y_clipped(int index) const { 
        auto v = data[TRI_IDX(index) + 1]; 
        return Vec3(v.x(), v.y(), v.z());
    }
    CPT_CPU_GPU Vec3 z_clipped(int index) const { 
        auto v = data[TRI_IDX(index) + 2]; 
        return Vec3(v.x(), v.y(), v.z());
    }

    CPT_CPU_GPU Vec3 get_sphere_point(const Vec3& normal, int index) const { 
        auto v = data[TRI_IDX(index)]; 
        return normal * v.w() + Vec3(v.x(), v.y(), v.z());      // center + radius * direction
    }
};

#ifdef USE_SOA
    template<typename InnerType>
    using ArrayType = SoA3<InnerType>;
#else
    template<typename InnerType>
    using ArrayType = AoS3<InnerType>;
#endif  // USE_AOS

using ConstF4Ptr   = const float4* const;
using ConstVertPtr = const PrecomputeAoS* const;
using ConstNormPtr = const ArrayType<Vec3>* const;
using ConstUVPtr   = const ArrayType<Vec2>* const;
