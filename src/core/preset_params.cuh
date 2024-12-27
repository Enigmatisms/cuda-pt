#pragma once
/**
 * @brief Metal Parameters & Dispersion Params
 */
#include <array>
#include "core/vec3.cuh"

enum MetalType: uint8_t {
    Au   = 0,
    Cr   = 1,
    Cu   = 2,
    Ag   = 3,
    Al   = 4,
    W    = 5,
    TiO2 = 6,
    Ni   = 7,
    MgO  = 8,
    Na   = 9,
    SiC  = 10,
    V    = 11,
    CuO  = 12,
    Hg   = 13,
    Ir   = 14,
    NumMetalType
};

inline constexpr std::array<const char*, NumMetalType> METAL_NAMES = {
    "Au", "Cr", "Cu", "Ag", "Al", "W", "TiO2", "Ni", "MgO", "Na", "SiC", "V", "CuO", "Hg", "Ir"
};

// Data from Tungsten Renderer
inline constexpr Vec3 METAL_ETA_TS[NumMetalType] = {
    Vec3(0.1431189557f, 0.3749570432f, 1.4424785571f),      // Au
    Vec3(4.3696828663f, 2.9167024892f, 1.6547005413f),      // Cr
    Vec3(0.2004376970f, 0.9240334304f, 1.1022119527f),      // Cu
    Vec3(0.1552646489f, 0.1167232965f, 0.1383806959f),      // Ag
    Vec3(1.6574599595f, 0.8803689579f, 0.5212287346f),      // Al
    Vec3(4.3707029924f, 3.3002972445f, 2.9982666528f),      // W
    Vec3(3.4566203131f, 2.8017076558f, 2.9051485020f),      // TiO2
    Vec3(2.3672753521f, 1.6633583302f, 1.4670554172f),      // Ni
    Vec3(2.0895885542f, 1.6507224525f, 1.5948759692f),      // MgO
    Vec3(0.0602665320f, 0.0561412435f, 0.0619909494f),      // Na 
    Vec3(3.1723450205f, 2.5259677964f, 2.4793623897f),      // SiC
    Vec3(4.2775126218f, 3.5131538236f, 2.7611257461f),      // V  
    Vec3(3.2453822204f, 2.4496293965f, 2.1974114493f),      // CuO
    Vec3(2.3989314904f, 1.4400254917f, 0.9095512090f),      // Hg
    Vec3(3.0864098394f, 2.0821938440f, 1.6178866805f)       // Ir
};

inline constexpr Vec3 METAL_KS[NumMetalType] = {
    Vec3(3.9831604247f, 2.3857207478f, 1.6032152899f),      // Au
    Vec3(5.2064337956f, 4.2313645277f, 3.7549467933f),      // Cr
    Vec3(3.9129485033f, 2.4528477015f, 2.1421879552f),      // Cu
    Vec3(4.8283433224f, 3.1222459278f, 2.1469504455f),      // Ag
    Vec3(9.2238691996f, 6.2695232477f, 4.8370012281f),      // Al
    Vec3(3.5006778591f, 2.6048652781f, 2.2731930614f),      // W
    Vec3(0.0001026662f, -0.0000897534f, 0.0006356902f),     // TiO2
    Vec3(4.4988329911f, 3.0501643957f, 2.3454274399),       // Ni
    Vec3(0.0000000000f, 0.0000000000f, 0.0000000000f),      // MgO
    Vec3(3.1792906496f, 2.1124800781f, 1.5790940266f),      // Na 
    Vec3(0.0000007284f, -0.0000006859f, 0.0000100150f),     // SiC
    Vec3(3.4911844504f, 2.8893580874f, 3.1116965117f),      // V  
    Vec3(0.5202739621f, 0.5707372756f, 0.7172250613f),      // CuO
    Vec3(6.3276269444f, 4.3719414152f, 3.4217899270f),      // Hg
    Vec3(5.5921510077f, 4.0671757150f, 3.2672611269f)       // Ir
};

// Data from Wikipedia and LuxRenderer
enum DispersionType: uint8_t {
    Diamond     = 0,
    Silica      = 1,
    Glass_BK7   = 2,
    Glass_BaF10 = 3,
    Glass_SF10  = 4,
    NumDispersionType
};

inline constexpr std::array<const char*, DispersionType::NumDispersionType> DISPERSION_NAMES = {
    "Diamond", "Silica", "Glass_BK7", "Glass_BaF10", "Glass_SF10"
};

inline constexpr Vec2 DISPERSION_PARAMS[NumDispersionType] = {
    Vec2(2.3840, 34221.2),     // Diamond
    Vec2(1.4580, 3540.0),      // Silica
    Vec2(1.5046, 4200.0),      // Glass_BK7
    Vec2(1.6700, 7430.0),      // Glass_BaF10
    Vec2(1.7280, 13420.0)      // Glass_SF10
};