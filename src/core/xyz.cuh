/**
 * @file xyz.cuh
 * @author Qianyue He
 * @brief XYZ space conversion
 * @date 2024-12-27
 * @copyright Copyright (c) 2024
 */
#pragma once
#include "core/vec3.cuh"
#include "core/vec4.cuh"

class ColorSpaceXYZ {
private:
    cudaArray_t _CIE_data;
    cudaArray_t _D65_data;

    void to_gpu() const;
public:
    cudaTextureObject_t CIE;       // float4
    cudaTextureObject_t D65;       // float
    static constexpr int CIE_samples = 471;
    static constexpr int D65_samples = 531;
public:
    void init();
    void destroy();

    /**
     * Check: https://en.wikipedia.org/wiki/SRGB#Correspondence_to_CIE_XYZ_stimulus
     */
    CONDITION_TEMPLATE(VecType, Vec4)
    CPT_CPU_GPU_INLINE static Vec4 XYZ_to_sRGB(VecType&& XYZ) {
        float r = 3.240479f  * XYZ.x() + -1.537150f * XYZ.y() + -0.498535f * XYZ.z();
        float g = -0.969256f * XYZ.x() + 1.875991f  * XYZ.y() + 0.041556f  * XYZ.z();
        float b = 0.055648f  * XYZ.x() + -0.204043f * XYZ.y() + 1.057311f  * XYZ.z();
        return Vec4(r, g, b);
    }
};

template <typename TexType>
static cudaTextureObject_t createArrayTexture1D(
    const TexType* tex_src, 
    cudaArray_t& arr_t,
    size_t size 
) {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<TexType>();
    CUDA_CHECK_RETURN(cudaMallocArray(&arr_t, &channel_desc, size));
    CUDA_CHECK_RETURN(cudaMemcpy2DToArray(arr_t, 0, 0, tex_src, size * sizeof(TexType), size * sizeof(TexType), 1, cudaMemcpyHostToDevice));

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = arr_t;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = true;

    cudaTextureObject_t tex_obj;
    CUDA_CHECK_RETURN(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));
    return tex_obj;
}

extern CPT_GPU_CONST ColorSpaceXYZ XYZ;