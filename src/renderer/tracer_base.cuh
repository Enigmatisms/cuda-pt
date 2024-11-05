/**
 * Base class of path tracers
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/scene.cuh"
#include "core/host_device.cuh"
#include "renderer/base_pt.cuh"

class TracerBase {
protected:
    AABB* aabbs;
    PrecomputedArray verts;
    ArrayType<Vec3> norms; 
    ConstBuffer<PackedHalf2> uvs;
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
        const Scene& scene
    ): verts(scene.num_prims), norms(scene.num_prims), uvs(scene.num_prims),
       image(scene.config.width, scene.config.height), 
       num_prims(scene.num_prims), 
       w(scene.config.width), 
       h(scene.config.height)
    {
        scene.export_prims(verts, norms, uvs);
        CUDA_CHECK_RETURN(cudaMallocManaged(&aabbs, num_prims * sizeof(AABB)));
        ShapeAABBVisitor aabb_visitor(verts, aabbs);
        // calculate AABB for each primitive
        for (int i = 0; i < num_prims; i++) {
            aabb_visitor.set_index(i);
            std::visit(aabb_visitor, scene.shapes[i]);
        }
    }

    ~TracerBase() {
        CUDA_CHECK_RETURN(cudaFree(aabbs));
        image.destroy();
        verts.destroy();
        norms.destroy();
        uvs.destroy();
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
