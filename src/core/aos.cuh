/**
 * Easy SoA (Struct of Arrays) encapsulation
 * SoA can facilitate coalesced memory access
 * and cache coherence
 *
 * SoA3 aims to hold three different field in a array
 * for example, vec3 (x, y, z)
*/

#pragma once
#include "core/vec4.cuh"
#include "core/defines.cuh"
#include "core/vec2_half.cuh"
#include "core/host_device.cuh"

#define TRI_IDX(index) (index << 1) + index

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
        if (size > 0) {
            size = 0;
            CUDA_CHECK_RETURN(cudaFree(data));
        }
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

    CPT_CPU_GPU_INLINE const StructType& x(int index) const { return data[index]; }
    CPT_CPU_GPU_INLINE const StructType& y(int index) const { return data[index + size]; }
    CPT_CPU_GPU_INLINE const StructType& z(int index) const { return data[index + (size << 1)]; }

    CPT_CPU_GPU_INLINE StructType& x(int index) { return data[index]; }
    CPT_CPU_GPU_INLINE StructType& y(int index) { return data[index + size]; }
    CPT_CPU_GPU_INLINE StructType& z(int index) { return data[index + (size << 1)]; }

    CPT_GPU_INLINE StructType eval(int index, float u, float v) const { return StructType{}; }
};

template<>
CPT_GPU_INLINE Vec3 SoA3<Vec3>::eval(int index, float u, float v) const { 
    return (data[index] * (1.f - u - v) + \
            data[index + size] * u + \
            data[index + (size << 1)] * v).normalized();
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
        if (size > 0) {
            size = 0;
            CUDA_CHECK_RETURN(cudaFree(data));
        }
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

    CPT_CPU_GPU_INLINE Vec4 x(int index) const { return data[INDEX_X(index, size)]; }
    CPT_CPU_GPU_INLINE Vec4 y(int index) const { return data[INDEX_Y(index, size)]; }
    CPT_CPU_GPU_INLINE Vec4 z(int index) const { return data[INDEX_Z(index, size)]; }

    CPT_CPU_GPU_INLINE float2 x_front(int index_2) const { return (reinterpret_cast<const float2*>(data))[(index_2 << 1) + index_2]; }
    CPT_CPU_GPU_INLINE float2 y_front(int index_2) const { return (reinterpret_cast<const float2*>(data))[(index_2 << 1) + index_2 + 2]; }
    CPT_CPU_GPU_INLINE float2 z_front(int index_2) const { return (reinterpret_cast<const float2*>(data))[(index_2 << 1) + index_2 + 4]; }

    CPT_CPU_GPU_INLINE float2 x_back(int index_2) const { return (reinterpret_cast<const float2*>(data))[(index_2 << 1) + index_2 + 1]; }
    CPT_CPU_GPU_INLINE float2 y_back(int index_2) const { return (reinterpret_cast<const float2*>(data))[(index_2 << 1) + index_2 + 3]; }
    CPT_CPU_GPU_INLINE float2 z_back(int index_2) const { return (reinterpret_cast<const float2*>(data))[(index_2 << 1) + index_2 + 5]; }

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

class NormalAoSArray {
private:
    cudaTextureObject_t tex_obj;
    float4* data;         // packed sphere data or vertex difference (with adjoint matrix terms packed)
    size_t pitch;
public:
    static constexpr int pad_to_2048(int num) {
        return num > 2048 ? (num + 2047) & ~2047 : num;
    }

    CPT_CPU NormalAoSArray(size_t _size) {
        // lazy initialization
    }

    CPT_CPU void destroy() {
        if (data != nullptr) {
            CUDA_CHECK_RETURN(cudaDestroyTextureObject(tex_obj));
            CUDA_CHECK_RETURN(cudaFree(data));
            data = nullptr;
        }
    }

    // GPU implements SoA constructor very differently
    CPT_GPU NormalAoSArray(float4* _data, cudaTextureObject_t obj) :
        data(_data), tex_obj(obj) {}

    // vertices input, convert to float4 and store precomputed adjoint matrix value
    // convert from vertices to vertex difference
    CPT_CPU void from_vectors(
        const std::vector<Vec3>& vec1,
        const std::vector<Vec3>& vec2,
        const std::vector<Vec3>& vec3,
        const std::vector<bool>* const sphere_flags = nullptr         // use naked ptr to enable null
    ) {
        size_t height = pad_to_2048(vec1.size()) >> 11, width = 0;
        if (height > 0) {           // size equals to or exceeds 2048
            width = 3 * 2048;
        } else {                    // size less than 2048
            width = 3 * vec1.size();
            height = 1;
        }

        float4* temp_data = nullptr;
        const size_t host_pitch = width * sizeof(float4);
        CUDA_CHECK_RETURN(cudaMallocHost(&temp_data, host_pitch * height));
        for (size_t i = 0; i < vec1.size(); i++) {
            Vec3 v1 = vec1[i], v2 = vec2[i], v3 = vec3[i];
            temp_data[TRI_IDX(i) + 0] = make_float4(v1.x(), v1.y(), v1.z(), 0);
            temp_data[TRI_IDX(i) + 1] = make_float4(v2.x(), v2.y(), v2.z(), 0);
            temp_data[TRI_IDX(i) + 2] = make_float4(v3.x(), v3.y(), v3.z(), 0);
        }

        CUDA_CHECK_RETURN(cudaMallocPitch(&data, &pitch, width * sizeof(float4), height));
        CUDA_CHECK_RETURN(cudaMemcpy2D(data, pitch, temp_data, host_pitch, host_pitch, height, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaFreeHost(temp_data));

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = data;
        res_desc.res.pitch2D.desc = channel_desc;
        res_desc.res.pitch2D.width = width;
        res_desc.res.pitch2D.height = height;
        res_desc.res.pitch2D.pitchInBytes = pitch;

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeClamp;      
        tex_desc.addressMode[1] = cudaAddressModeClamp;      
        tex_desc.filterMode     = cudaFilterModePoint;           
        tex_desc.readMode       = cudaReadModeElementType;         
        tex_desc.normalizedCoords = 0;              
        CUDA_CHECK_RETURN(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));
    }

    CPT_GPU_INLINE Vec3 eval(int index, float u, float v) const {
        float y = index >> 11;       // index divided by 2048
        index %= 2048;                  // width
        // Vec4 lerp_v = 
        //     Vec4(tex2D<float4>(tex_obj, float(TRI_IDX(index) + 0) + (1.f - u - v) / (1.f - v), y)) * (1.f - v) +
        //     Vec4(tex2D<float4>(tex_obj, float(TRI_IDX(index) + 2), y)) * v;
        // return Vec3(lerp_v.xyz()).normalized();
        auto n1 = Vec4(tex2D<float4>(tex_obj, TRI_IDX(index) + 0, y)),
             n2 = Vec4(tex2D<float4>(tex_obj, TRI_IDX(index) + 1, y)),
             n3 = Vec4(tex2D<float4>(tex_obj, TRI_IDX(index) + 2, y));
        return (Vec3(n1.xyz()) * (1.f - u - v) + Vec3(n2.xyz()) * u + Vec3(n3.xyz()) * v).normalized();
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

template<typename InnerType>
using ArrayType = SoA3<InnerType>;

#undef INDEX_X
#undef INDEX_Y
#undef INDEX_Z

using ConstU4Ptr   = const uint4* const __restrict__;
using ConstVertPtr = const PrecomputedArray* const __restrict__;
using ConstUVPtr   = const ConstBuffer<PackedHalf2>* const __restrict__;

#ifdef USE_TEX_NORMAL
    using NormalArray  = NormalAoSArray;
#else
    using NormalArray  = ArrayType<Vec3>;
#endif

