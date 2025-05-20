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

#pragma once

#define TRIANGLE_ONLY     // Use only triangles as primitives (10% faster for
                          // massive triangle only scene)
#define NO_ORTHOGONAL_CAM // Disable orthogonal camera
#define USE_TEX_NORMAL    // Use texture memory to hold the vertex normals
#define SUPPORTS_TOF_RENDERING // Whether to support ToF rendering

// #define NO_RAY_SORTING              // WFPT: disable ray sorting according to
// material ID

// #define NO_STREAM_COMPACTION        // Disable stream compaction
// #define STABLE_PARTITION            // Use thrust::stable_partition to
// maintain the order #define FUSED_MISS_SHADER           // Whether to use
// fused miss shader (in closest hit shader)

#ifdef SUPPORTS_TOF_RENDERING
#define CONDITION_BLOCK(x) if (x)
#else
#define CONDITION_BLOCK(x) if constexpr (true)
#endif // SUPPORTS_TOF_RENDERING
