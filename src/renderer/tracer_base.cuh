/**
 * Base class of path tracers
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/aos.cuh"
#include "core/shapes.cuh"
#include "core/host_device.cuh"
#include "core/camera_model.cuh"

extern __constant__ DeviceCamera dev_cam;

// #define CP_BASE_6
#ifdef CP_BASE_6
static constexpr int BASE_SHFL = 6;
using BitMask = uint64_t;
CPT_GPU_INLINE int __count_bit(BitMask bits) { return __ffsll(bits); } 
#else
static constexpr int BASE_SHFL = 5;
using BitMask = uint32_t;
CPT_GPU_INLINE int __count_bit(BitMask bits) { return __ffs(bits); } 
#endif
static constexpr int BASE_ADDR = 1 << BASE_SHFL;

/**
 * This API is deprecated, due to the performance bounded by BSYNC
 * which is the if branch barrier synchronization (convergence problem)
 * 
 * Take a look at the stackoverflow post I posted:
 * https://stackoverflow.com/questions/78603442/convergence-barrier-for-branchless-cuda-conditional-select
*/
CPT_GPU float ray_intersect_old(
    const Ray& ray,
    ConstShapePtr shapes,
    ConstAABBWPtr s_aabbs,
    ShapeIntersectVisitor& shape_visitor,
    int& min_index,
    const int remain_prims,
    const int cp_base_5,
    float min_dist
) {
    float aabb_tmin = 0; 
    #pragma unroll
    for (int idx = 0; idx < remain_prims; idx ++) {
        if (s_aabbs[idx].aabb.intersect(ray, aabb_tmin) && aabb_tmin <= min_dist) {
            shape_visitor.set_index(idx);
            float dist = variant::apply_visitor(shape_visitor, shapes[cp_base_5 + idx]);
            bool valid = dist > EPSILON && dist < min_dist;
            min_dist = valid ? dist : min_dist;
            min_index = valid ? cp_base_5 + idx : min_index;
        }
    }
    return min_dist;
}

/**
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
 * compare to the ray_intersect_old, this API almost double the speed
*/
CPT_GPU float ray_intersect(
    const Ray& ray,
    ConstShapePtr shapes,
    ConstAABBWPtr s_aabbs,
    ShapeIntersectVisitor& shape_visitor,
    int& min_index,
    const int remain_prims,
    const int cp_base,
    float min_dist
) {
    float aabb_tmin = 0;
    BitMask tasks = 0;          // 32 bytes

#pragma unroll
    for (int idx = 0; idx < remain_prims; idx++) {
        // if current ray intersects primitive at [idx], tasks will store it
        BitMask valid_intr = s_aabbs[idx].aabb.intersect(ray, aabb_tmin) && (aabb_tmin < min_dist);
        tasks |= (valid_intr << (BitMask)idx);
        // note that __any_sync here won't work well
    }
#pragma unroll
    while (tasks) {
        BitMask idx = __count_bit(tasks) - 1; // find the first bit that is set to 1, note that __ffs is 
        tasks &= ~((BitMask)1 << idx); // clear bit in case it is found again
        shape_visitor.set_index(idx);
        float dist = variant::apply_visitor(shape_visitor, shapes[cp_base + idx]);
        bool valid = dist > EPSILON && dist < min_dist;
        min_dist = valid ? dist : min_dist;
        min_index = valid ? cp_base + idx : min_index;
    }
     return min_dist;
}

class TracerBase {
protected:
    Shape* shapes;
    AABB* aabbs;
    ArrayType<Vec3>* verts;
    ArrayType<Vec3>* norms; 
    ArrayType<Vec2>* uvs;
    DeviceImage image;
    DeviceImage* dev_image;
    int num_prims;
    int w;
    int h;
public:
    /**
     * @param shapes    shape information (for ray intersection)
     * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
    */
    TracerBase(
        const std::vector<Shape>& _shapes,
        const ArrayType<Vec3>& _verts,
        const ArrayType<Vec3>& _norms, 
        const ArrayType<Vec2>& _uvs,
        int width, int height
    ): image(width, height), dev_image(nullptr),
       num_prims(_shapes.size()), w(width), h(height)
    {
        CUDA_CHECK_RETURN(cudaMallocManaged(&shapes, num_prims * sizeof(Shape)));
        CUDA_CHECK_RETURN(cudaMallocManaged(&aabbs, num_prims * sizeof(AABB)));
        ShapeAABBVisitor aabb_visitor(_verts, aabbs);
        // calculate AABB for each primitive
        for (int i = 0; i < num_prims; i++) {
            shapes[i] = _shapes[i];
            aabb_visitor.set_index(i);
            variant::apply_visitor(aabb_visitor, _shapes[i]);
        }

        dev_image = to_gpu(image);
        verts = to_gpu(_verts);
        norms = to_gpu(_norms);
        uvs   = to_gpu(_uvs);
    }

    ~TracerBase() {
        CUDA_CHECK_RETURN(cudaFree(shapes));
        CUDA_CHECK_RETURN(cudaFree(aabbs));

        CUDA_CHECK_RETURN(cudaFree(dev_image));
        CUDA_CHECK_RETURN(cudaFree(verts));
        CUDA_CHECK_RETURN(cudaFree(norms));
        CUDA_CHECK_RETURN(cudaFree(uvs));
    }

    CPT_CPU virtual std::vector<uint8_t> render(
        int num_iter  = 64,
        int max_depth = 1,/* max depth, useless for depth renderer, 1 anyway */
        bool gamma_correction = true
    ) {
        throw std::runtime_error("Not implemented.\n");
        return {};
    }
};
