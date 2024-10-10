/**
 * Base class of path tracers
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/aos.cuh"
#include "core/bsdf.cuh"
#include "core/object.cuh"
#include "core/host_device.cuh"
#include "core/camera_model.cuh"
#include "core/shapes.cuh"
#include "core/shapes.cuh"
#include "renderer/base_pt.cuh"

class TracerBase {
protected:
    AABB* aabbs;
    PrecomputedArray* verts;
    ArrayType<Vec3>* norms; 
    ArrayType<Vec2>* uvs;
    DeviceImage image;
    int num_prims;
    int w;
    int h;
public:
    /**
     * @param shapes    shape information (for AABB generation)
     * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
    */
    TracerBase(
        const std::vector<Shape>& _shapes,
        const PrecomputedArray& _verts,
        const ArrayType<Vec3>& _norms, 
        const ArrayType<Vec2>& _uvs,
        int width, int height
    ): image(width, height), num_prims(_shapes.size()), w(width), h(height)
    {
        CUDA_CHECK_RETURN(cudaMallocManaged(&aabbs, num_prims * sizeof(AABB)));
        ShapeAABBVisitor aabb_visitor(_verts, aabbs);
        // calculate AABB for each primitive
        for (int i = 0; i < num_prims; i++) {
            aabb_visitor.set_index(i);
            std::visit(aabb_visitor, _shapes[i]);
        }

        verts = to_gpu(_verts);
        norms = to_gpu(_norms);
        uvs   = to_gpu(_uvs);
    }

    ~TracerBase() {
        CUDA_CHECK_RETURN(cudaFree(aabbs));
        CUDA_CHECK_RETURN(cudaFree(verts));
        CUDA_CHECK_RETURN(cudaFree(norms));
        CUDA_CHECK_RETURN(cudaFree(uvs));
        image.destroy();
    }

    CPT_CPU virtual std::vector<uint8_t> render(
        int num_iter  = 64,
        int max_depth = 1,/* max depth, useless for depth renderer, 1 anyway */
        bool gamma_correction = true
    ) {
        throw std::runtime_error("Not implemented.\n");
        return {};
    }

    CPT_CPU virtual void render_online(
        int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
    ) {
        throw std::runtime_error("Not implemented.\n");
    }
};
