/**
 * Easy SoA (Struct of Arrays) encapsulation
 * SoA can facilitate coalesced memory access
 * and cache coherence
 *
 * SoA3 aims to hold three different field in a array
 * for example, vec3 (x, y, z)
*/

#pragma once
#include "core/vec2_half.cuh"
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

// AOS (this is actually faster)
#define INDEX_X(index, size) TRI_IDX(index) + 0
#define INDEX_Y(index, size) TRI_IDX(index) + 1
#define INDEX_Z(index, size) TRI_IDX(index) + 2
// SOA (this is actually slower)
// #define INDEX_X(index, size) index
// #define INDEX_Y(index, size) index + size
// #define INDEX_Z(index, size) index + (size << 1)

// PrecomputedArray stored triangle difference, two of the adjacent matrix terms
// and one Vec3 array for absolute positioning (of a triangle)
// ray intersection and emitter sampling efficiency can be boosted
// 9 floats per triangle -> 12 floats per triangle
// This class can be both SoA and AoS, depending on the actual compilation setting
class PrecomputedArray {
public:
    Vec4* data;         // packed sphere data or vertex difference (with adjoint matrix terms packed)
    size_t size;
public:
    CPT_CPU PrecomputedArray(size_t _size) : size(_size) {
        CUDA_CHECK_RETURN(cudaMallocManaged(&data, 3 * sizeof(Vec4) * _size));
    }

    CPT_CPU void destroy() {
        size = 0;
        CUDA_CHECK_RETURN(cudaFree(data));
    }

    // GPU implements SoA constructor very differently
    CPT_GPU PrecomputedArray(Vec4* _data, size_t size) :
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
                data[INDEX_X(i, size)] = Vec4(center.x(), center.y(), center.z(), vec2[i].x());
                data[INDEX_Y(i, size)] = Vec4();
                data[INDEX_Z(i, size)] = Vec4();
                continue;
            }
            Vec3 abs1  = vec1[i],
                 diff1 = vec2[i] - abs1,
                 diff2 = vec3[i] - abs1;
            // precomputed adjacent matrix terms, work as padding (saved 3 FMADD, 3 FMUL and 3 FADD per triangle)
            float a20 = diff1.y() * diff2.z() - diff1.z() * diff2.y(),
                  a21 = diff2.x() * diff1.z() - diff1.x() * diff2.z(),
                  a22 = diff1.x() * diff2.y() - diff2.x() * diff1.y();
            data[INDEX_X(i, size)] = Vec4(abs1.x(), abs1.y(), abs1.z(), a20);
            data[INDEX_Y(i, size)] = Vec4(diff1.x(), diff1.y(), diff1.z(), a21);
            data[INDEX_Z(i, size)] = Vec4(diff2.x(), diff2.y(), diff2.z(), a22);
        }
    }

    CPT_CPU void fill(const Vec4& v) {
        parallel_memset << <1, 256 >> > (data, v, size * 3);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }

    CPT_CPU_GPU_INLINE const Vec4& x(int index) const { return data[INDEX_X(index, size)]; }
    CPT_CPU_GPU_INLINE const Vec4& y(int index) const { return data[INDEX_Y(index, size)]; }
    CPT_CPU_GPU_INLINE const Vec4& z(int index) const { return data[INDEX_Z(index, size)]; }

    CPT_CPU_GPU_INLINE Vec4& x(int index) { return data[INDEX_X(index, size)]; }
    CPT_CPU_GPU_INLINE Vec4& y(int index) { return data[INDEX_Y(index, size)]; }
    CPT_CPU_GPU_INLINE Vec4& z(int index) { return data[INDEX_Z(index, size)]; }

    CPT_CPU_GPU_INLINE Vec3 x_clipped(int index) const { 
        auto v = data[INDEX_X(index, size)]; 
        return Vec3(v.x(), v.y(), v.z());
    }
    CPT_CPU_GPU_INLINE Vec3 y_clipped(int index) const { 
        auto v = data[INDEX_Y(index, size)]; 
        return Vec3(v.x(), v.y(), v.z());
    }
    CPT_CPU_GPU_INLINE Vec3 z_clipped(int index) const { 
        auto v = data[INDEX_Z(index, size)]; 
        return Vec3(v.x(), v.y(), v.z());
    }

    CPT_CPU_GPU_INLINE Vec3 get_sphere_point(const Vec3& normal, int index) const { 
        auto v = data[INDEX_Z(index, size)]; 
        return normal * v.w() + Vec3(v.x(), v.y(), v.z());      // center + radius * direction
    }
};

// ConstBuffer won't be destroyed, unless we manually call `destroy`
template <typename Ty>
class ConstBuffer {
private:
    Ty* _buffer;
    size_t _size;
public:
    CPT_CPU ConstBuffer(size_t _sz): _size(_sz) {
        if (_size <= 0) {
            throw std::runtime_error("Allocated buffer size is not a positive value.\n");
        }
        CUDA_CHECK_RETURN(cudaMallocManaged(&_buffer, sizeof(Ty) * _sz));
    }

    CPT_CPU ConstBuffer(const Ty* const inputs, size_t _sz): _size(_sz) {
        if (_size <= 0) {
            throw std::runtime_error("Allocated buffer size is not a positive value.\n");
        }
        CUDA_CHECK_RETURN(cudaMallocManaged(&_buffer, sizeof(Ty) * _sz));
        CUDA_CHECK_RETURN(cudaMemcpy(_buffer, inputs, sizeof(Ty) * _sz, cudaMemcpyHostToDevice));
    }

    CPT_CPU bool destroy() {
        if (_size != 0) {
            CUDA_CHECK_RETURN(cudaFree(_buffer));
            _size = 0;
            return true;
        }
        return false;
    }

    // setting the buffer can only occur in host side   
    CPT_CPU_INLINE Ty& operator[](int index) { return _buffer[index]; }
    CPT_CPU_INLINE Ty* data() noexcept { return _buffer; }

    CPT_CPU_GPU_INLINE const Ty& operator[](int index) const { return _buffer[index]; }
    CPT_CPU_GPU_INLINE const Ty* data() const noexcept { return _buffer; }
    CPT_CPU_GPU_INLINE size_t size() const noexcept { return _size; }
};

#ifdef USE_SOA
    template<typename InnerType>
    using ArrayType = SoA3<InnerType>;
#else
    template<typename InnerType>
    using ArrayType = AoS3<InnerType>;
#endif  // USE_AOS

#undef INDEX_X
#undef INDEX_Y
#undef INDEX_Z

using ConstF4Ptr   = const float4* const __restrict__;
using ConstVertPtr = const PrecomputedArray* const __restrict__;
using ConstNormPtr = const ArrayType<Vec3>* const __restrict__;
using ConstUVPtr   = const ConstBuffer<PackedHalf2>* const __restrict__;