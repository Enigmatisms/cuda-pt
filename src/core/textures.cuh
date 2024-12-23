#pragma once
/**
 * @brief Support for CUDA texture. Online loading is not supported
 * @author Qianyue He
 * @date   2024.12.24
 */

#include "core/vec2.cuh"
#include "core/vec3.cuh"
#include "core/vec4.cuh"

// TODO: we need host side Texture support

// This should be stored as SoA
class Textures {
private:
    // The underlying texture data
    float4** float4_ptrs;
    float3** float3_ptrs;
    float2** float2_ptrs;
public:
    cudaTextureObject_t* diff_tex;      // float4 (TBA, alpha channel is never used)
    cudaTextureObject_t* spec_tex;      // float4
    cudaTextureObject_t* glos_tex;      // float4
    cudaTextureObject_t* roughness;     // float2
    cudaTextureObject_t* normals;       // float3
public:
    CPT_GPU_INLINE Vec4 evaluate_4d(const cudaTextureObject_t& target, float u, float v) const {
        return Vec4(tex2D<float4>(target, u, v));
    }

    CPT_GPU_INLINE Vec3 evaluate_3d(const cudaTextureObject_t& target, float u, float v) const {
        return Vec3(tex2D<float3>(target, u, v));
    }

    CPT_GPU_INLINE Vec2 evaluate_2d(const cudaTextureObject_t& target, float u, float v) const {
        return Vec2(tex2D<float2>(target, u, v));
    }

    Textures(const std::vector<cudaTextureObject_t>& tex_objs) {
        const size_t total_size = sizeof(cudaTextureObject_t) * tex_objs.size() * 5;
        CUDA_CHECK_RETURN(cudaMalloc(&diff_tex, total_size));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(diff_tex, tex_objs.data(), total_size, cudaMemcpyHostToDevice));
        spec_tex  = diff_tex + tex_objs.size();
        glos_tex  = spec_tex + tex_objs.size();
        roughness = glos_tex + tex_objs.size();
        normals   = roughness + tex_objs.size();
    }

    ~Textures() {
        CUDA_CHECK_RETURN(cudaFree(diff_tex));
    }
};

