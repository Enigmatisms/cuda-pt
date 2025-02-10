/**
 * Some general constants
 * @author: Qianyue He
 * @date:   5.12.2024
*/

#pragma once

constexpr float EPSILON = 1e-3f;
constexpr float THP_EPS  = 1e-4f;
constexpr float AABB_EPS = 1e-3f;
constexpr float MAX_DIST = 1e7;
constexpr float ENVMAP_DIST = 5e3;
constexpr float AABB_INVALID_DIST = 1e5;
constexpr float SCALING_EPS = 1.05f;

constexpr float M_Pi = 3.1415926535897f;
constexpr float M_2Pi = M_Pi * 2;
constexpr float M_1_Pi = 1.f / M_Pi;
constexpr float DEG2RAD = M_Pi / 180.f;

constexpr int INVALID_OBJ = 0xffffffff;