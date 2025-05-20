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
 * @file enums.cuh
 * @author Qianyue He
 * @version 0.1
 * @date 2025-01-09
 * @copyright Copyright (c) 2025
 */
#pragma once
#include <stdint.h>

enum RendererType : uint8_t {
    MegaKernelPT = 0,  // megakernel path tracing
    WavefrontPT = 1,   // wavefront  path tracing
    MegaKernelLT = 2,  // megakernel light tracing
    VoxelSDFPT = 3,    // not supported currently
    DepthTracing = 4,  // rendering depth map
    BVHCostViz = 5,    // displaying BVH traversal cost
    MegaKernelVPT = 6, // megakernel volumetric path tracer
    NumRendererType
};

enum ScatterStateFlag : int {
    BSDF_NONE = 0x00,
    BSDF_DIFFUSE = 0x01,
    BSDF_SPECULAR = 0x02,
    BSDF_GLOSSY = 0x04,
    BSDF_FORWARD = 0x08,

    BSDF_REFLECT = 0x10,
    BSDF_TRANSMIT = 0x20,
    SCAT_VOLUME = 0x40,
};

enum BSDFType : uint8_t {
    Lambertian = 0x00,
    Specular = 0x01,
    Translucent = 0x02,
    Plastic = 0x03,
    PlasticForward = 0x04,
    GGXConductor = 0x05,
    Dispersion = 0x06,
    Forward = 0x07,
    NumSupportedBSDF = 0x08
};

enum PhaseFuncType : uint8_t {
    NullForward = 0x00,
    Isotropic = 0x01,
    HenyeyGreenstein = 0x02,
    DuoHG = 0x03,
    Rayleigh = 0x04,
    SGGX = 0x05,
    NumSupportedPhase = 0x06
};

enum MediumType : uint8_t {
    Homogeneous = 0x00,
    Grid = 0x01,
    NumSupportedMedium = 0x02
};

enum TextureType : uint8_t {
    DIFFUSE_TEX = 0x0,
    SPECULAR_TEX = 0x1,
    GLOSSY_TEX = 0x2,
    NORMAL_TEX = 0x3,
    ROUGHNESS_TEX = 0x4,
};

enum MetalType : uint8_t {
    Au = 0,
    Cr = 1,
    Cu = 2,
    Ag = 3,
    Al = 4,
    W = 5,
    TiO2 = 6,
    Ni = 7,
    MgO = 8,
    Na = 9,
    SiC = 10,
    V = 11,
    CuO = 12,
    Hg = 13,
    Ir = 14,
    NumMetalType
};

// Data from Wikipedia and LuxRenderer
enum DispersionType : uint8_t {
    Diamond = 0,
    DiamondHigh = 1,
    Silica = 2,
    Glass_BK7 = 3,
    Glass_BaF10 = 4,
    Glass_SF10 = 5,
    Sapphire = 6,
    Water = 7,
    NumDispersionType
};
