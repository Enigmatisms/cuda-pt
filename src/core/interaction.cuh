/**
 * Ray intersection (or scattering event) interaction point
 * @author: Qianyue He
 * @date:   4.29.2024
*/
#pragma once
#include "core/vec2_half.cuh"

// Now, this data type can fit in a 16 Byte float4 (128 bit L/S is possible)
class Interaction {
public:
    Vec3 shading_norm;      // size: 3 floats
    Vec2Half uv_coord;      // size: 1 float

    CPT_CPU_GPU Interaction() {}

    template <typename Vec3Type, typename Vec2Type>
    CPT_CPU_GPU Interaction(Vec3Type&& _n, Vec2Type&& _uv): 
        shading_norm(std::forward<Vec3Type>(_n)), 
        uv_coord(std::forward<Vec2Type>(_uv))
    {
        static_assert(
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<Vec3Type>>, Vec3> &&
            (std::is_same_v<std::remove_cv_t<std::remove_reference_t<Vec2Type>>, Vec2Half> ||
             std::is_same_v<std::remove_cv_t<std::remove_reference_t<Vec2Type>>, Vec2>),
            "Input type check failed"
        );
    }
};