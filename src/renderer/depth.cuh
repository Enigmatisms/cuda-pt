/**
 * Simple tile-based depth renderer
 * @date: 5.6.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include "core/stats.h"
#include "renderer/tracer_base.cuh"

// #define CP_BASE_6
#ifdef CP_BASE_6
constexpr int BASE_SHFL = 6;
using BitMask = long long;
CPT_GPU_INLINE int __count_bit(BitMask bits) { return __ffsll(bits); } 
#else
constexpr int BASE_SHFL = 5;
using BitMask = int;
CPT_GPU_INLINE int __count_bit(BitMask bits) { return __ffs(bits); } 
#endif
constexpr int BASE_ADDR = 1 << BASE_SHFL;

/**
 * Perform ray-intersection test on shared memory primitives
 * @param s_verts:      vertices stored in shared memory
 * @param ray:          the ray for intersection test
 * @param s_aabbs:      scene primitives
 * @param remain_prims: number of primitives to be tested (32 at most)
 * @param min_index:    closest hit primitive index
 * @param min_obj_idx:  closest hit object index
 * @param prim_u:       intersection barycentric coord u
 * @param prim_v:       intersection barycentric coord v
 * @param min_dist:     current minimal distance
 *
 * @return minimum intersection distance
 * 
 * compare to the ray_intersect_old, this API almost double the speed
 * To check how I improve the naive ray intersection, see Commit: a6602786f9b1a4e70036288a6778c7fcb0b0f75b 
*/
CPT_GPU float ray_intersect(
    const PrecomputedArray& s_verts, 
    const Ray& ray,
    ConstAABBWPtr s_aabbs,
    const int remain_prims,
    const int cp_base,
    int& min_index,
    int& min_obj_idx,
    float& prim_u,
    float& prim_v,
    float min_dist
) {
    float aabb_tmin = 0;
    BitMask tasks = 0;

#pragma unroll
    for (int idx = 0; idx < remain_prims; idx ++) {
        // if current ray intersects primitive at [idx], tasks will store it
        BitMask valid_intr = s_aabbs[idx].aabb.intersect(ray, aabb_tmin) && aabb_tmin < min_dist;
        tasks |= valid_intr << (BitMask)idx;
    }
#pragma unroll
    while (tasks) {
        BitMask idx = __count_bit(tasks) - 1; // find the first bit that is set to 1, note that __ffs is 
        tasks &= ~((BitMask)1 << idx); // clear bit in case it is found again
        int obj_idx = s_aabbs[idx].aabb.obj_idx();
#ifdef TRIANGLE_ONLY
        float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, s_verts, idx, it_u, it_v, true);
#else
        float it_u = 0, it_v = 0, dist = Primitive::intersect(ray, s_verts, idx, it_u, it_v, obj_idx >= 0);
#endif
        bool valid = dist > EPSILON && dist < min_dist;
        min_dist = valid ? dist : min_dist;
        prim_u   = valid ? it_u : prim_u;
        prim_v   = valid ? it_v : prim_v;
        min_index = valid ? cp_base + idx : min_index;
        min_obj_idx = valid ? obj_idx : min_obj_idx;
    }
    return min_dist;
}

/**
 * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
 * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
 * @param camera    GPU camera model (constant memory)
 * @param image     GPU image buffer
 * @param num_prims number of primitives (to be intersected with)
 * @param max_depth maximum allowed bounce
*/
CPT_KERNEL static void render_depth_kernel(
    const DeviceCamera& dev_cam,
    const PrecomputedArray verts,
    const NormalArray norms, 
    const ConstBuffer<PackedHalf2> uvs,
    ConstAABBPtr aabbs,
    DeviceImage image,
    int num_prims,
    int max_bounce = 1/* max depth, useless for depth renderer, 1 anyway */
) {
    int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    Sampler sampler(px + py * image.w());
    // step 1: generate ray
    Ray ray = dev_cam.generate_ray(px, py, sampler);

    // step 2: bouncing around the scene until the max depth is reached
    Interaction it;

    // A matter of design choice
    // optimization: copy at most 32 prims from global memory to shared memory

    __shared__ Vec4 s_verts[TRI_IDX(32)];         // vertex info
    __shared__ AABBWrapper s_aabbs[32];            // aabb
    PrecomputedArray s_verts_arr(reinterpret_cast<Vec4*>(&s_verts[0]), BASE_ADDR);

    int num_copy = (num_prims + 31) / 32;   // round up
    float min_dist = MAX_DIST;
    for (int b = 0; b < max_bounce; b++) {
        for (int cp_base = 0; cp_base < num_copy; ++cp_base) {
            // memory copy to shared memory
            int cp_base_5 = cp_base << 5, cur_idx = cp_base_5 + tid, remain_prims = min(num_prims - cp_base_5, 32);
            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
            if (tid < 32 && cur_idx < num_prims) {        // copy from gmem to smem
                cuda::memcpy_async(&s_verts[TRI_IDX(tid)], &verts.data[TRI_IDX(cur_idx)], sizeof(Vec4) * 3, pipe);
                s_aabbs[tid].aabb.copy_from(aabbs[cur_idx]);
            }
            pipe.producer_commit();
            pipe.consumer_wait();
            __syncthreads();
            // step 3: traverse all the primitives and intersect

            int min_index = 0, object_id = 0;
            float prim_u = 0, prim_v = 0;
            min_dist = ray_intersect(s_verts_arr, ray, s_aabbs, remain_prims, 
                cp_base << BASE_SHFL, min_index, object_id, prim_u, prim_v, min_dist);
            __syncthreads();
        }
    }
    image(px, py) += min_dist * (min_dist < MAX_DIST);
}

class DepthTracer: public TracerBase {
using TracerBase::verts;
using TracerBase::norms; 
using TracerBase::uvs;
using TracerBase::image;
using TracerBase::num_prims;
using TracerBase::w;
using TracerBase::h;

DeviceCamera* camera;
AABB* aabbs;
public:
    DepthTracer(const Scene& scene): TracerBase(scene) {
        CUDA_CHECK_RETURN(cudaMalloc(&camera, sizeof(DeviceCamera)));
        CUDA_CHECK_RETURN(cudaMemcpy(camera, &scene.cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMallocManaged(&aabbs, num_prims * sizeof(AABB)));
        ShapeAABBVisitor aabb_visitor(verts, aabbs);
        // calculate AABB for each primitive
        for (int i = 0; i < num_prims; i++) {
            aabb_visitor.set_index(i);
            std::visit(aabb_visitor, scene.shapes[i]);
        }
    }

    ~DepthTracer() {
        CUDA_CHECK_RETURN(cudaFree(camera));
    }

    CPT_CPU std::vector<uint8_t> render(
        const MaxDepthParams& md,
        int num_iter = 64,
        bool gamma_correction = true
    ) override {
        printf("Depth tracing. Num primitives: %d, max_depth: %d\n", num_prims, md.max_depth);
        TicToc _timer("render_kernel()", num_iter);
        for (int i = 0; i < num_iter; i++) {
            // for more sophisticated renderer (like path tracer), shared_memory should be used
            render_depth_kernel<<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
                    *camera, verts, norms, uvs, aabbs, image, num_prims, md.max_depth); 
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }
        return image.export_cpu(1.f / (15.f * num_iter), gamma_correction);
    }


};

