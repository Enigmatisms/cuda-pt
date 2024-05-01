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
    bool valid;

    Interaction(): shading_norm(), uv_coord(), valid(false) {}

    template <typename Vec3Type, typename Vec2Type>
    Interaction(Vec3Type&& _n, Vec2Type&& _uv, bool _v = false): 
        shading_norm(std::forward<Vec3Type>(_n)), 
        uv_coord(std::forward<Vec2Type>(_uv)), valid(_v) 
    {
        static_assert(
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<Vec3Type>>, Vec3> &&
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<Vec2Type>>, Vec2>,
            "Input type check failed"
        );
    }
};