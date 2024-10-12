/**
 * Light tracing for caustics rendering
 * @date: 9.28.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include <cuda_gl_interop.h>
#include "core/stats.h"
#include "core/scene.cuh"
#include "core/progress.h"
#include "renderer/path_tracer.cuh"

class LightTracer: public PathTracer {
private:
    bool bidirectional;         // whether to use both PT and LT in a single renderer
    int spec_constraint;
    float caustic_scaling;
public:
    /**
     * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
     * 
     * @todo: initialize emitters
     * @todo: initialize objects
    */
    LightTracer(
        const Scene& scene,
        const PrecomputedArray& _verts,
        const ArrayType<Vec3>& _norms, 
        const ArrayType<Vec2>& _uvs,
        int num_emitter,
        int spec_constraint,
        bool bidir = false,
        float caustics_scale = 1.f
    ): PathTracer(scene, _verts, _norms, _uvs, num_emitter), 
        spec_constraint(spec_constraint), 
        bidirectional(bidir),
        caustic_scaling(caustics_scale) {}

    virtual CPT_CPU std::vector<uint8_t> render(
        int num_iter = 64,
        int max_depth = 4,
        bool gamma_correction = true
    ) override;

    virtual CPT_CPU void render_online(
        int max_depth = 4
    ) override;
};
