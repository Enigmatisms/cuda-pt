/**
 * Megakernel Path Tracing
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include "core/bvh.cuh"
#include "core/bsdf.cuh"
#include "core/shapes.cuh"
#include "core/emitter.cuh"
#include "core/camera_model.cuh"

extern __constant__ Emitter* c_emitter[9];          // c_emitter[8] is a dummy emitter
extern __constant__ BSDF*    c_material[32];

using ConstBVHPtr  = const LinearBVH* const;
using ConstNodePtr = const LinearNode* const;
using ConstObjPtr   = const ObjInfo* const;
using ConstBSDFPtr  = const BSDF* const;
using ConstIndexPtr = const int* const;

/**
 * Occlusion test, computation is done on global memory
*/
CPT_GPU bool occlusion_test(
    const Ray& ray,
    ConstObjPtr objects,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    const ArrayType<Vec3>& verts,
    int num_objects,
    float max_dist
);

// occlusion test is any hit shader
CPT_GPU bool occlusion_test_bvh(
    const Ray& ray,
    ConstShapePtr shapes,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    const ArrayType<Vec3>& verts,
    const int node_num,
    float max_dist
);

/**
 * Stackless BVH (should use tetxure memory?)
 * Perform ray-intersection test on shared memory primitives
 * @param ray: the ray for intersection test
 * @param shapes: scene primitives
 * @param s_aabbs: scene primitives
 * @param shape_visitor: encapsulated shape visitor
 * @param it: interaction info, containing the interacted normal and uv
 * @param remain_prims: number of primitives to be tested (32 at most)
 * @param cp_base: shared memory address offset
 * @param min_dist: current minimal distance
 *
 * @return minimum intersection distance
 * 
 * ray_intersect_bvh is closesthit shader
 * compare to the ray_intersect_old, this API almost double the speed
*/
CPT_GPU float ray_intersect_bvh(
    const Ray& ray,
    ConstShapePtr shapes,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    ShapeIntersectVisitor& shape_visitor,
    int& min_index,
    const int node_num,
    float min_dist
);

CPT_GPU Emitter* sample_emitter(Sampler& sampler, float& pdf, int num, int no_sample);

/**
 * @brief this version does not employ object-level culling
 * we use shared memory to accelerate rendering instead, for object-level culling
 * shared memory might not be easy to use, since the memory granularity will be
 * too difficult to control
 * 
 * @param objects   object encapsulation
 * @param prim2obj  primitive to object index mapping: which object does this primitive come from?
 * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
 * @param camera    GPU camera model (constant memory)
 * @param image     GPU image buffer
 * @param num_prims number of primitives (to be intersected with)
 * @param max_depth maximum allowed bounce
*/
template <bool render_once>
CPT_KERNEL static void render_pt_kernel(
    const DeviceCamera& dev_cam, 
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    cudaTextureObject_t bvh_fronts,
    cudaTextureObject_t bvh_backs,
    cudaTextureObject_t node_fronts,
    cudaTextureObject_t node_backs,
    cudaTextureObject_t node_offsets,
    DeviceImage& image,
    float* output_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int seed_offset,
    int max_depth = 1,/* max depth, useless for depth renderer, 1 anyway */
    int node_num = -1,
    int accum_cnt = 1
);