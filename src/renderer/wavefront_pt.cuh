/**
 * (W)ave(f)ront (S)imple path tracing with stream multiprocessing
 * We first consider how to make a WF tracer, then we start optimizing it, hence this is a 'Simple' one 
 * 
 * for each stream, we will create their own ray pools for
 * stream compaction and possible execution reordering
 * 
 * each stream contains 4 * 4 blocks, each block contains 16 * 16 threads, which is therefore
 * a 64 * 64 pixel patch. We will only create at most 8 streams, to fill up the host-device connections
 * therefore, it is recommended that the image sizes are the multiple of 64
 * 
 * for each kernel function, sx (int) and sy (int) are given, which is the base location of the current
 * stream. For example, let there be 4 streams and 4 kernel calls and the image is of size (256, 256)
 * stream 1: (0, 0), (64, 0), (128, 0), (192, 0)                |  1   2   3   4  |
 * stream 2: (0, 64), (64, 64), (128, 64), (192, 64)            |  1   2   3   4  |
 * stream 3: (0, 128), (64, 128), (128, 128), (192, 128)        |  1   2   3   4  |
 * stream 4: (0, 192), (64, 192), (128, 192), (192, 192)        |  1   2   3   4  |
 * 
 * @author Qianyue He
 * @date   2024.6.20
*/
#pragma once
#include <omp.h>
#include <cuda/pipeline>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "core/progress.h"
#include "core/emitter.cuh"
#include "core/object.cuh"
#include "core/scene.cuh"
#include "core/stats.h"
#include "renderer/path_tracer.cuh"

// #define NO_STREAM_COMPACTION
#define STABLE_PARTITION
#define NO_RAY_SORTING
#define FUSED_MISS_SHADER

#ifdef STABLE_PARTITION
#define partition_func(...) thrust::stable_partition(__VA_ARGS__)
#else
#define partition_func(...) thrust::partition(__VA_ARGS__)
#endif    // STABLE_PARTITION

namespace {
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;
    constexpr int THREAD_X = 16;
    constexpr int THREAD_Y = 16;
    constexpr int PATCH_X = BLOCK_X * THREAD_X;
    constexpr int PATCH_Y = BLOCK_Y * THREAD_Y;
    constexpr int TOTAL_RAY = PATCH_X * PATCH_Y;

    using IndexBuffer = uint32_t* const;
}

union PDFInteraction {
    struct {
        float pdf;
        Interaction it;
    } v;
    struct {
        float4 p1;
        float2 p2;
    } data;

    CPT_CPU_GPU PDFInteraction() {}
    CPT_CPU_GPU_INLINE PDFInteraction(float pdf) { v.pdf = pdf; }

    CPT_CPU_GPU_INLINE Interaction& it() { return this->v.it; }    
    CPT_CPU_GPU_INLINE const Interaction& it_const() const { return this->v.it; }    

    CPT_CPU_GPU_INLINE float& pdf() { return this->v.pdf; }    
    CPT_CPU_GPU_INLINE float pdf_v() const { return this->v.pdf; }    

};

class PayLoadBufferSoA {
public:
    struct RayOrigin {
        Vec3 o;
        float hit_t;
        CPT_CPU_GPU_INLINE RayOrigin() {}

        CPT_CPU_GPU_INLINE RayOrigin(const Ray& ray) { FLOAT4(o) = CONST_FLOAT4(ray.o); }
        CPT_CPU_GPU_INLINE void get(Ray& ray) const { FLOAT4(ray.o) = CONST_FLOAT4(o); }
        CPT_CPU_GPU_INLINE void set(const Ray& ray) { FLOAT4(o) = CONST_FLOAT4(ray.o); }
    };

    struct RayDirTag {
        Vec3 d;
        uint32_t ray_tag;
        CPT_CPU_GPU_INLINE RayDirTag() {}

        CPT_CPU_GPU_INLINE RayDirTag(const Ray& ray) { FLOAT4(d) = CONST_FLOAT4(ray.d); }
        CPT_CPU_GPU_INLINE void get(Ray& ray) const { FLOAT4(ray.d) = CONST_FLOAT4(d); }
        CPT_CPU_GPU_INLINE void set(const Ray& ray) { FLOAT4(d) = CONST_FLOAT4(ray.d); }

        CPT_CPU_GPU_INLINE void set_active(bool v) noexcept {
            ray_tag &= 0xefffffff;      // clear bit 28
            ray_tag |= uint32_t(v) << 28;
        }

        CPT_CPU_GPU_INLINE bool is_hit() const noexcept {
            return (ray_tag & 0x20000000) > 0;
        }
    };
private:
    template <typename VectorType>
    struct Length4Data {
        VectorType data;
        CPT_CPU_GPU_INLINE Length4Data() {}
    };
    template <typename VectorType>
    struct Length2Data {
        VectorType data;
        CPT_CPU_GPU_INLINE Length2Data() {}
    };
public:

    Vec4* thps;             // direct malloc
    Vec4* Ls;               // direct malloc
    
    RayOrigin* ray_os;
    RayDirTag* ray_ds;
    Length4Data<uint4>* samp_heads;
    Length2Data<uint2>* samp_tails;
    Length4Data<float4>* its_heads;
    Length2Data<float2>* its_tails;

    int    _width;       

    CPT_CPU_GPU PayLoadBufferSoA() {}

    CPT_CPU void init(int width, int height) {
        _width = width;
        int full_size = width * height;
        CUDA_CHECK_RETURN(cudaMalloc(&thps, sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&Ls,   sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&ray_os, sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&ray_ds, sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&samp_heads, sizeof(Length4Data<uint4>) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&its_heads,  sizeof(Length4Data<float4>) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&samp_tails, sizeof(Length2Data<uint2>) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&its_tails,  sizeof(Length2Data<float2>) * full_size));
    }

    CPT_CPU_GPU_INLINE Vec4& thp(int col, int row) { return thps[col + row * _width]; }
    CPT_CPU_GPU_INLINE const Vec4& thp(int col, int row) const { return thps[col + row * _width]; }

    CPT_CPU_GPU_INLINE Vec4& L(int col, int row) { return Ls[col + row * _width]; }
    CPT_CPU_GPU_INLINE const Vec4& L(int col, int row) const { return Ls[col + row * _width]; }

    CONDITION_TEMPLATE(RayType, Ray)
    CPT_CPU_GPU_INLINE void set_ray(int col, int row, RayType&& ray) { 
        int index = col + row * _width;
        ray_os[index].set(ray);
        ray_ds[index].set(ray);
    }

    CPT_CPU_GPU_INLINE Ray get_ray(int col, int row) const { 
        Ray ray;
        int index = col + row * _width;
        ray_os[index].get(ray);
        ray_ds[index].get(ray);
        return ray; 
    }

    CPT_CPU_GPU_INLINE void set_active(int col, int row, bool v) noexcept {
        int index = col + row * _width;
        ray_ds[index].set_active(v);
    }

    CPT_CPU_GPU_INLINE bool is_hit(int col, int row) const noexcept {
        int index = col + row * _width;
        return ray_ds[index].is_hit();
    }

    CPT_CPU_GPU_INLINE bool is_active(int col, int row) const noexcept {
        return (ray_ds[col + row * _width].ray_tag & 0x10000000) > 0;
    }

    CPT_CPU_GPU_INLINE Sampler get_sampler(int col, int row) const { 
        static_assert(std::is_same_v<Sampler, TinySampler>);
        Sampler samp;
        int index = col + row * _width;
        UINT4(samp._get_v_front()) = samp_heads[index].data;
        UINT2(samp._get_d_front()) = samp_tails[index].data;
        return samp; 
    }

    CONDITION_TEMPLATE(SamplerType, TinySampler)
    CPT_CPU_GPU_INLINE void set_sampler(int col, int row, SamplerType&& sampler) { 
        int index = col + row * _width;
        samp_heads[index].data = CONST_UINT4(sampler._get_v_front());
        samp_tails[index].data = CONST_UINT2(sampler._get_d_front());
    }

    CPT_CPU_GPU_INLINE PDFInteraction get_interaction(int col, int row) const { 
        PDFInteraction it;
        int index = col + row * _width;
        it.data.p1 = its_heads[index].data;
        it.data.p2 = its_tails[index].data;
        return it; 
    }

    CONDITION_TEMPLATE(ItType, PDFInteraction)
    CPT_CPU_GPU_INLINE void set_interaction(int col, int row, ItType&& it) { 
        int index = col + row * _width;
        its_heads[index].data = it.data.p1;
        its_tails[index].data = it.data.p2;
    }

    CPT_CPU_GPU_INLINE void set_it_head(int col, int row, float4 p1) { 
        int index = col + row * _width;
        its_heads[index].data = p1;
    }

    CPT_CPU void destroy() {
        CUDA_CHECK_RETURN(cudaFree(thps));
        CUDA_CHECK_RETURN(cudaFree(Ls));
        CUDA_CHECK_RETURN(cudaFree(ray_os));
        CUDA_CHECK_RETURN(cudaFree(ray_ds));
        CUDA_CHECK_RETURN(cudaFree(samp_heads));
        CUDA_CHECK_RETURN(cudaFree(samp_tails));
        CUDA_CHECK_RETURN(cudaFree(its_heads));
        CUDA_CHECK_RETURN(cudaFree(its_tails));
    }
};

/**
 * @brief ray generation kernel 
 * note that, all the kernels are called per stream, each stream can have multiple blocks (since it is a kernel call)
 * let's say, for example, a 4 * 4 block for one kernel call. These 16 blocks should be responsible for 
 * one image patch, offseted by the stream_offset.
 * @note we first consider images that have width and height to be the multiple of 128
 * to avoid having to consider the border problem
 * @note we pass payloads in by value
*/ 
CPT_KERNEL void raygen_primary_hit_shader(
    const DeviceCamera& dev_cam,
    PayLoadBufferSoA payloads,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    const IndexBuffer idx_buffer,
    int stream_offset, int num_prims,
    int x_patch, int y_patch, int iter,
    int stream_id, int width, int node_num = -1
);

/**
 * @brief find ray intersection for next hit pos
 * We first start with small pool size (4096), which can comprise at most 16 blocks
 * The ray pool is stream-compacted (with thrust::parition to remove the finished)
 * Note that we need an index buffer, since the Ray and Sampler are coupled
 * and we need the index to port the 
*/ 
CPT_KERNEL void closesthit_shader(
    PayLoadBufferSoA payloads,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims,
    int num_valid,
    int node_num = -1
);

/***
 * For non-delta hit (shading point), direct component should be evaluated:
 * we sample a light source then start ray intersection test
*/
CPT_KERNEL void nee_shader(
    PayLoadBufferSoA payloads,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr,         
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims,
    int num_objects,
    int num_emitter,
    int num_valid,
    int node_num = -1
);


/**
 * BSDF sampling & direct shading shader
*/
CPT_KERNEL void bsdf_local_shader(
    PayLoadBufferSoA payloads,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstUVPtr,         
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims, 
    int num_valid,
    bool secondary_bounce
);

/**
 * Purpose of the miss shader: if ray hits nothing in closesthit shader
 * the we will set the hit status (flag) to be false
 * in this shader, we find the rays marked as no-hit, and check the
 * availability of environment map (currently not supported)
 * after processing the env-map lighting, we mark the ray as inactive
 * before stream compaction. Then stream compaction will 'remove' all these
 * rays (and the threads)
 * 
 * MISS_SHADER is the only place where you mark a ray as inactive
*/
CPT_KERNEL void miss_shader(
    PayLoadBufferSoA payloads,
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_valid
);

CPT_KERNEL void radiance_splat(
    PayLoadBufferSoA payloads, DeviceImage& image, 
    int stream_id, int x_patch, int y_patch
);

/**
 * This functor is used for stream compaction
*/
struct ActiveRayFunctor
{
    const int width;
    const PayLoadBufferSoA::RayDirTag* const dir_tags;

    CPT_CPU_GPU ActiveRayFunctor(PayLoadBufferSoA::RayDirTag* tags, int w): width(w), dir_tags(tags) {}

    CPT_GPU_INLINE bool operator()(uint32_t index) const
    {
        uint32_t py = index >> 16, px = index & 0x0000ffff;
        return (dir_tags[px + py * width].ray_tag & 0x10000000) > 0;
    }
};


/**
 * This functor is used for ray index sorting
*/
struct RaySortFunctor
{
    const int width;
    const PayLoadBufferSoA::RayDirTag* const dir_tags;

    CPT_CPU_GPU RaySortFunctor(PayLoadBufferSoA::RayDirTag* tags, int w): width(w), dir_tags(tags) {}

    CPT_GPU_INLINE bool operator()(uint32_t idx1, uint32_t idx2) const
    {
        return (dir_tags[(idx1 & 0x0000ffff) + (idx1 >> 16) * width].ray_tag & 0x0fffffff) <
               (dir_tags[(idx2 & 0x0000ffff) + (idx2 >> 16) * width].ray_tag & 0x0fffffff);
    }
};