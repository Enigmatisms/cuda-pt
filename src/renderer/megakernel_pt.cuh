/**
 * Megakernel Path Tracing
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/max_depth.h"
#include "core/camera_model.cuh"
#include "renderer/tracing_func.cuh"

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level culling
 * shared memory might not be easy to use, since the memory granularity will be
 * too difficult to control
 * 
 * @param objects        object encapsulation
 * @param verts          vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms          normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs            uv coordinates, Packed 3 Half2 and 1 int for padding (sum up to 128 bits)
 * @param emitter_prims  Primitive indices for emission objects
 * @param bvh_leaves     BVH leaf nodes (int texture, storing primitive to obj index map)
 * @param nodes          BVH nodes (32 Bytes)
 * @param cached_nodes   BVH cached nodes (in shared memory): first half: front float4, second half: back float4
 * @param image          GPU image buffer
 * @param output_buffer  Possible visualization buffer
 * @param num_emitter    number of emitters
 * @param seed_offset    offset to random seed (to create uncorrelated samples)
 * @param md_params      maximum allowed bounces (total, diffuse, specular, transmission)
 * @param node_num       number of nodes on a BVH tree
 * @param accum_cnt      Counter of iterations
 * @param cache_num      Number of cached BVH nodes
 * @param gamma_corr     For online rendering, whether to enable gamma correction on visualization
*/
template <bool render_once>
CPT_KERNEL void render_pt_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const NormalArray norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    const MaxDepthParams md_params,
    float* __restrict__ output_buffer,
    float* __restrict__ var_buffer,
    int num_emitter,
    int seed_offset,
    int node_num  = -1,
    int accum_cnt = 1,
    int cache_num = 0,
    int envmap_id = 0,
    bool gamma_corr = false
);

/**
 * Megakernel Light Tracing. Light tracing is only used to render
 * complex caustics: starting from the emitter, we will only record
 * path which has more than `specular_constraints` number of
 * specular nodes
 * @param specular_constraints The path throughput will be ignored
 * if number of specular events is less or equal to this value
*/
template <bool render_once>
CPT_KERNEL void render_lt_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const NormalArray norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    const MaxDepthParams md_params,
    float* __restrict__ output_buffer,
    int num_emitter,
    int seed_offset,
    int node_num = -1,
    int accum_cnt = 1,
    int cache_num = 0,
    int specular_constraints = 0,
    float caustic_scale = 1.f,
    bool gamma_corr = false
);