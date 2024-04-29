/**
 * Ray intersection (or scattering event) interaction point
 * @author: Qianyue He
 * @date:   4.29.2024
*/
#pragma once
#include "core/vec2.cuh"

template <typename Ty>
class Interaction {
public:
    Vec3<Ty> shading_norm;
    Vec2<Ty> uv_coord;
};