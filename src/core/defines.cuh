#pragma once

#define TRIANGLE_ONLY               // Use only triangles as primitives (10% faster for massive triangle only scene)
#define NO_ORTHOGONAL_CAM           // Disable orthogonal camera
#define USE_TEX_NORMAL              // Use texture memory to hold the vertex normals

// #define NO_RAY_SORTING              // WFPT: disable ray sorting according to material ID

// #define NO_STREAM_COMPACTION        // Disable stream compaction
// #define STABLE_PARTITION            // Use thrust::stable_partition to maintain the order
// #define FUSED_MISS_SHADER           // Whether to use fused miss shader (in closest hit shader)