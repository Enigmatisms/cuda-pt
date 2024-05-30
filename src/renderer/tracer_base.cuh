/**
 * Base class of path tracers
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/soa.cuh"
#include "core/shapes.cuh"
#include "core/host_device.cuh"
#include "core/camera_model.cuh"

extern __constant__ DeviceCamera dev_cam;

/**
 * Perform ray-intersection test on shared memory primitives
 * @param ray: the ray for intersection test
 * @param shapes: scene primitives
 * @param s_aabbs: scene primitives
 * @param shape_visitor: encapsulated shape visitor
 * @param it: interaction info, containing the interacted normal and uv
 * @param remain_prims: number of primitives to be tested (32 at most)
 * @param cp_base_5: shared memory address offset
 * @param min_dist: current minimal distance
 * 
 * @return minimum intersection distance
*/
CPT_GPU float ray_intersect(
    const Ray& ray,
    ConstShapePtr shapes,
    ConstAABBPtr s_aabbs,
    ShapeIntersectVisitor& shape_visitor,
    int& min_index,
    int remain_prims,
    int cp_base_5,
    float min_dist
) {
    float aabb_tmin = 0;
    #pragma unroll
    for (int idx = 0; idx < remain_prims; idx ++) {
        if (s_aabbs[idx].intersect(ray, aabb_tmin) && aabb_tmin <= min_dist) {
            shape_visitor.set_index(idx);
            float dist = variant::apply_visitor(shape_visitor, shapes[cp_base_5 + idx]);
            bool valid = dist > EPSILON;
            valid &= dist < min_dist;       // whether to update the distance
            min_dist = dist * valid + min_dist * (1 - valid);
            min_index = (cp_base_5 + idx) * valid + min_index * (1 - valid);
            // printf("AABB intersect (%d): %f, %f, ray: (%f, %f, %f), (%f, %f, %f)\n", remain_prims, dist, min_dist, ray.o.x(), ray.o.y(), ray.o.z(), ray.d.x(), ray.d.y(), ray.d.z());
        }
    }
    return min_dist;
}

class TracerBase {
protected:
    Shape* shapes;
    AABB* aabbs;
    SoA3<Vec3>* verts;
    SoA3<Vec3>* norms; 
    SoA3<Vec2>* uvs;
    DeviceImage image;
    DeviceImage* dev_image;
    int num_prims;
    int w;
    int h;
public:
    /**
     * @param shapes    shape information (for ray intersection)
     * @param verts     vertices, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, SoA3: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param uvs       uv coordinates, SoA3: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
    */
    TracerBase(
        const std::vector<Shape>& _shapes,
        const SoA3<Vec3>& _verts,
        const SoA3<Vec3>& _norms, 
        const SoA3<Vec2>& _uvs,
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
