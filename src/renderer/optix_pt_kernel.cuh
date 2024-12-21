/**
 * Megakernel Path Tracing
 * @date: 9.15.2024
 * @author: Qianyue He
*/
#pragma once
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include "core/bsdf.cuh"
#include "core/object.cuh"
#include "core/emitter.cuh"
#include "core/camera_model.cuh"

#define RENDERER_USE_BVH

extern __constant__ Emitter* c_emitter[9];          // c_emitter[8] is a dummy emitter
extern __constant__ BSDF*    c_material[32];

using ConstBSDFPtr  = const BSDF* const __restrict__;
using ConstIndexPtr = const int* const __restrict__;

struct LaunchParams {
    // actually, there can be more than this member, but for now, I don't need any
    OptixTraversableHandle handle;
};

extern "C" {
    extern __constant__ LaunchParams params;
}

/**
 * @brief Megakernel Path tracing kernel function with optixTrace
 */
template <bool render_once>
CPT_KERNEL void render_optix_kernel(
    const DeviceCamera& dev_cam, 
    const PrecomputedArray verts,
    const ArrayType<Vec3> norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstObjPtr objects,
    const cudaTextureObject_t obj_idxs,
    ConstIndexPtr emitter_prims,
    DeviceImage image,
    float* __restrict__ output_buffer,
    int num_emitter,
    int seed_offset,
    int max_depth,
    int accum_cnt,
    bool gamma_corr
);