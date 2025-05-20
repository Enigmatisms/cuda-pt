// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * @author Qianyue He
 * @brief (W)ave(f)ront (S)imple path tracing with stream multiprocessing
 * We first consider how to make a WF tracer, then we start optimizing it, hence
 * this is a 'Simple' one
 *
 * for each stream, we will create their own ray pools for
 * stream compaction and possible execution reordering
 *
 * each stream contains 4 * 4 blocks, each block contains 16 * 16 threads, which
 * is therefore a 64 * 64 pixel patch. We will only create at most 8 streams, to
 * fill up the host-device connections therefore, it is recommended that the
 * image sizes are the multiple of 64
 *
 * for each kernel function, sx (int) and sy (int) are given, which is the base
 * location of the current stream. For example, let there be 4 streams and 4
 * kernel calls and the image is of size (256, 256) stream 1: (0, 0), (64, 0),
 * (128, 0), (192, 0)                |  1   2   3   4  | stream 2: (0, 64), (64,
 * 64), (128, 64), (192, 64)            |  1   2   3   4  | stream 3: (0, 128),
 * (64, 128), (128, 128), (192, 128)        |  1   2   3   4  | stream 4: (0,
 * 192), (64, 192), (128, 192), (192, 192)        |  1   2   3   4  |
 *
 * @date   2024.6.20
 */
#pragma once
#include "core/emitter.cuh"
#include "core/object.cuh"
#include "core/progress.h"
#include "core/scene.cuh"
#include "core/stats.h"
#include "renderer/path_tracer.cuh"
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <omp.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>

// #define STABLE_PARTITION

#ifdef STABLE_PARTITION
#define partition_func(...) thrust::stable_partition(__VA_ARGS__)
#else
#define partition_func(...) thrust::partition(__VA_ARGS__)
#endif // STABLE_PARTITION

using IndexBuffer = uint32_t *const __restrict__;

/**
 * Ray Pool, 96 Bytes Per Ray, which means
 * 2880 * 1920 image: only less than half a GB
 *
 * So, accounting for index buffer (int) and ray tag buffer (for ray sorting,
 * int) Together with output buffer, only 102 Byte per Ray
 */
class PayLoadBufferSoA {
  public:
    struct RayOrigin {
        Vec3 o;
        float hit_t;
        CPT_CPU_GPU_INLINE RayOrigin() {}

        CPT_CPU_GPU_INLINE RayOrigin(const Ray &ray) {
            FLOAT4(o) = CONST_FLOAT4(ray.o);
        }
        CPT_CPU_GPU_INLINE void get(Ray &ray) const {
            FLOAT4(ray.o) = CONST_FLOAT4(o);
        }
        CPT_CPU_GPU_INLINE void set(const Ray &ray) {
            FLOAT4(o) = CONST_FLOAT4(ray.o);
        }
    };

    struct RayDirTag {
        Vec3 d;
        uint32_t ray_tag;
        CPT_CPU_GPU_INLINE RayDirTag() {}

        CPT_CPU_GPU_INLINE RayDirTag(const Ray &ray) {
            FLOAT4(d) = CONST_FLOAT4(ray.d);
        }
        CPT_CPU_GPU_INLINE void get(Ray &ray) const {
            FLOAT4(ray.d) = CONST_FLOAT4(d);
        }
        CPT_CPU_GPU_INLINE void set(const Ray &ray) {
            FLOAT4(d) = CONST_FLOAT4(ray.d);
        }

        CPT_CPU_GPU_INLINE void set_active(bool v) noexcept {
            ray_tag &= 0xefffffff; // clear bit 28
            ray_tag |= uint32_t(v) << 28;
        }

        CPT_CPU_GPU_INLINE bool is_active() const noexcept {
            return (ray_tag & 0x10000000) > 0;
        }

        CPT_CPU_GPU_INLINE bool is_hit() const noexcept {
            return (ray_tag & 0x20000000) > 0;
        }
    };

  public:
    // 6 float4 -> 96 Bytes in total
    Vec4 *thps; // (float4)
    Vec4 *Ls;   // (float4)

    RayOrigin *ray_os;         // (float4)
    RayDirTag *ray_ds;         // (float4)
    Interaction *interactions; // (float4)
    uint2 *samplers;
    // note that index buffer is excluded from PayLoadBufferSoA
    // but it takes up 4 Bytes, so

    float *pdfs;

    CPT_CPU_GPU PayLoadBufferSoA() {}

    CPT_CPU void init(int full_size) {
        CUDA_CHECK_RETURN(cudaMalloc(&thps, sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&Ls, sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&ray_os, sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&ray_ds, sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&interactions, sizeof(uint4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&samplers, sizeof(uint2) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&pdfs, sizeof(float) * full_size));
    }

    CPT_GPU_INLINE Vec4 &thp(int index) { return thps[index]; }
    CPT_GPU_INLINE const Vec4 &thp(int index) const { return thps[index]; }

    CPT_GPU_INLINE Vec4 &L(int index) { return Ls[index]; }
    CPT_GPU_INLINE const Vec4 &L(int index) const { return Ls[index]; }

    CONDITION_TEMPLATE(RayType, Ray)
    CPT_GPU_INLINE void set_ray(int index, RayType &&ray) {
        ray_os[index].set(ray);
        ray_ds[index].set(ray);
    }

    CPT_GPU_INLINE Ray get_ray(int index) const {
        Ray ray;
        ray_os[index].get(ray);
        ray_ds[index].get(ray);
        return ray;
    }

    CPT_GPU_INLINE void set_active(int index, bool v) noexcept {
        ray_ds[index].set_active(v);
    }

    CPT_GPU_INLINE bool is_hit(int index) const noexcept {
        return ray_ds[index].is_hit();
    }

    CPT_GPU_INLINE void get_ray_d(int index, Vec3 &dir,
                                  bool &is_active) const noexcept {
        auto temp = ray_ds[index];
        dir = temp.d;
        is_active = temp.is_active();
    }

    CPT_GPU_INLINE Sampler get_sampler(int index) const {
        Sampler samp;
        UINT2(samp._get_d_front()) = samplers[index];
        return samp;
    }

    CONDITION_TEMPLATE(SamplerType, Sampler)
    CPT_GPU_INLINE void set_sampler(int index, SamplerType &&sampler) {
        samplers[index] = CONST_UINT2(sampler._get_d_front());
    }

    CPT_GPU_INLINE Interaction interaction(int index) const {
        return interactions[index];
    }

    CPT_GPU_INLINE Interaction &interaction(int index) {
        return interactions[index];
    }

    CPT_GPU_INLINE float pdf(int index) const { return pdfs[index]; }

    CPT_GPU_INLINE float &pdf(int index) { return pdfs[index]; }

    CPT_CPU void destroy() {
        CUDA_CHECK_RETURN(cudaFree(thps));
        CUDA_CHECK_RETURN(cudaFree(Ls));
        CUDA_CHECK_RETURN(cudaFree(ray_os));
        CUDA_CHECK_RETURN(cudaFree(ray_ds));
        CUDA_CHECK_RETURN(cudaFree(interactions));
        CUDA_CHECK_RETURN(cudaFree(samplers));
        CUDA_CHECK_RETURN(cudaFree(pdfs));
    }
};

/**
 * @brief ray generation kernel, fusing a closest hit shader (for primary ray)
 */
CPT_KERNEL void raygen_primary_hit_shader(
    const DeviceCamera &dev_cam, PayLoadBufferSoA payloads,
    const PrecomputedArray verts, const NormalArray norms,
    const ConstBuffer<PackedHalf2> uvs, ConstObjPtr objects,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, IndexBuffer idx_buffer, int width, int node_num,
    int cache_num, int accum_cnt, int seed_offset, int envmap_id);

/**
 * @brief Fused shader: closest hit and miss shader
 * Except from raygen shader, all other shaders have very different shapes:
 * For example: <gridDim 1D, blockDim 1D>
 * gridDim: num_ray_payload / blockDim, blockDim = 128
 */
CPT_KERNEL void fused_closesthit_shader(
    PayLoadBufferSoA payloads, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, IndexBuffer idx_buffer, int node_num,
    int cache_num, int bounce, int envmap_id);

/***
 * Fusing NEE / Ray Scattering
 *
 */
CPT_KERNEL void fused_ray_bounce_shader(
    PayLoadBufferSoA payloads, const PrecomputedArray verts,
    const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects, ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves, const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes, const IndexBuffer idx_buffer, int num_emitter,
    int node_num, int cache_num, bool secondary_bounce);

template <bool render_once>
CPT_KERNEL void radiance_splat(PayLoadBufferSoA payloads, DeviceImage image,
                               float *__restrict__ output_buffer = nullptr,
                               float *__restrict__ var_buffer = nullptr,
                               int accum_cnt = 0, bool gamma_corr = false);

/**
 * @brief This shader is used in path guiding enabled WFPT
 * This kernel handles NEE and the ray hitting an emitter
 *
 * TODO: in this kernel, we must fill in the query
 * so that the neural network can eval the NASG params for us
 * before we execute the next kernel (`guided_ray_scatter_shader`)
 */
CPT_KERNEL void
nee_direct_shader(PayLoadBufferSoA payloads, const PrecomputedArray verts,
                  const NormalArray norms, const ConstBuffer<PackedHalf2> uvs,
                  ConstObjPtr objects, ConstIndexPtr emitter_prims,
                  const cudaTextureObject_t bvh_leaves,
                  const cudaTextureObject_t nodes, ConstF4Ptr cached_nodes,
                  const IndexBuffer idx_buffer, int num_emitter, int node_num,
                  int cache_num, bool secondary_bounce);

/**
 * @brief This kernel handles path guiding. Here
 * NASG [SIGGRAPH 2024] paper is reproduced (TODO)
 * For this kernel, we will wait until the neural network
 * outputs the evaluation. Note that the problem is:
 * (1) We should use multi-stream, to split up the evaluation
 * so that it won't be too memory-hungry, and this will also help
 * concurrency
 *
 * TODO: major refactoring here
 */
CPT_KERNEL void
guided_ray_scatter_net_eval_shader(PayLoadBufferSoA payloads,
                                   ConstObjPtr objects,
                                   const cudaTextureObject_t bvh_leaves,
                                   const IndexBuffer idx_buffer, int stream_id);

/**
 * This functor is used for stream compaction
 */
struct ActiveRayFunctor {
    CPT_GPU_INLINE bool operator()(uint32_t index) const {
        return index < 0x80000000;
    }
};
