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

extern __constant__ ColorSpaceXYZ XYZ;