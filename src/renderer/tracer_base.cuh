/**
 * Base class of path tracers
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include <variant/variant.h>
#include "core/soa.cuh"
#include "core/shapes.cuh"
#include "core/camera_model.cuh"
#include "core/host_device.cuh"
#include "core/stats.h"

extern __constant__ DeviceCamera dev_cam;

using Shape = variant::variant<TriangleShape, SphereShape>;
using ConstShapePtr = const Shape* const;

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
        int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
    ) {
        throw std::runtime_error("Not implemented.\n");
        return {};
    }
};
