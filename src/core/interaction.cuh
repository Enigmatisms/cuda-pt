/**
 * Ray intersection (or scattering event) interaction point
 * @author: Qianyue He
 * @date:   4.29.2024
*/
#pragma once
#include "core/vec2.cuh"

class Interaction {
public:
    Vec3 shading_norm;
    Vec2 uv_coord;

    CPT_CPU_GPU Interaction() {}

    template <typename Vec3Type, typename Vec2Type>
    CPT_CPU_GPU Interaction(Vec3Type&& _n, Vec2Type&& _uv): 
        shading_norm(std::forward<Vec3Type>(_n)), 
        uv_coord(std::forward<Vec2Type>(_uv))
    {
        static_assert(
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<Vec3Type>>, Vec3> &&
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<Vec2Type>>, Vec2>,
            "Input type check failed"
        );
    }
};