/**
 * Megakernel Path Tracing
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include "core/bvh.cuh"
#include "core/bsdf.cuh"
#include "core/primitives.cuh"
#include "core/emitter.cuh"
#include "core/camera_model.cuh"

#define RENDERER_USE_BVH

extern __constant__ Emitter* c_emitter[9];          // c_emitter[8] is a dummy emitter
extern __constant__ BSDF*    c_material[32];

using ConstNodePtr  = const LinearNode* const __restrict__;
using ConstObjPtr   = const ObjInfo* const __restrict__;
using ConstBSDFPtr  = const BSDF* const __restrict__;
using ConstIndexPtr = const int* const __restrict__;

/**
 * Occlusion test, computation is done on global memory
*/
CPT_GPU bool occlusion_test(
    const Ray& ray,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    const PrecomputedArray& verts,
    int num_objects,
    float max_dist
);

// occlusion test is any hit shader
CPT_GPU bool occlusion_test_bvh(
    const Ray& ray,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const PrecomputedArray& verts,
    const int node_num,
    const int cache_num,
    float max_dist
);

/**
 * Stackless BVH (should use tetxure memory?)
 * Perform ray-intersection test on shared memory primitives
*/
CPT_GPU float ray_intersect_bvh(
    const Ray& ray,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    const PrecomputedArray& verts,
    int& min_index,
    int& min_obj_idx,
    float& prim_u,
    float& prim_v,
    const int node_num,
    const int cache_num,
    float min_dist = MAX_DIST
);

CPT_GPU Emitter* sample_emitter(Sampler& sampler, float& pdf, int num, int no_sample);

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level culling
 * shared memory might not be easy to use, since the memory granularity will be
 * too difficult to control
 * 
 * @param objects       object encapsulation
 * @param verts         vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms         normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs           uv coordinates, Packed 3 Half2 and 1 int for padding (sum up to 128 bits)
 * @param emitter_prims Primitive indices for emission objects
 * @param bvh_leaves    BVH leaf nodes (int texture, storing primitive to obj index map)
 * @param nodes         BVH nodes (32 Bytes)
 * @param cached_nodes  BVH cached nodes (in shared memory): first half: front float4, second half: back float4
 * @param image         GPU image buffer
 * @param output_buffer Possible visualization buffer
 * @param num_prims     number of primitives (to be intersected with)
 * @param num_objects   number of objects
 * @param num_emitter   number of emitters
 * @param seed_offset   offset to random seed (to create uncorrelated samples)
 * @param max_depth     maximum allowed bounce
 * @param node_num      number of nodes on a BVH tree
 * @param accum_cnt     Counter of iterations
*/
template <bool render_once>
CPT_KERNEL void render_pt_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth = 1,
    int node_num  = -1,
    int accum_cnt = 1,
    int cache_num = 0
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
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    ConstAABBPtr aabbs,
    ConstIndexPtr emitter_prims,
    const cudaTextureObject_t bvh_leaves,
    const cudaTextureObject_t nodes,
    ConstF4Ptr cached_nodes,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth = 1,
    int node_num = -1,
    int accum_cnt = 1,
    int cache_num = 0,
    int specular_constraints = 0,
    float caustic_scale = 1.f
);