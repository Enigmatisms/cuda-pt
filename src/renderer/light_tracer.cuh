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
        const PrecomputeAoS& _verts,
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
    ) override {
        printf("Rendering starts.\n");
        TicToc _timer("render_lt_kernel()", num_iter);
        for (int i = 0; i < num_iter; i++) {
            // for more sophisticated renderer (like path tracer), shared_memory should be used
            if (bidirectional) {
                render_pt_kernel<true><<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
                    *camera, *verts, obj_info, aabbs, norms, uvs, 
                    bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets,
                    _cached_nodes, image, output_buffer, num_prims, num_objs, num_emitter, 
                    accum_cnt * SEED_SCALER, max_depth, num_nodes, accum_cnt
                ); 
                CUDA_CHECK_RETURN(cudaDeviceSynchronize());
            }
            render_lt_kernel<false><<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
                *camera, *verts, obj_info, aabbs, norms, uvs, 
                bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets,
                _cached_nodes, image, nullptr, num_prims, num_objs, num_emitter, 
                i * SEED_SCALER, max_depth, num_nodes, spec_constraint
            ); 
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
            printProgress(i, num_iter);
        }
        printf("\n");
        return image.export_cpu(1.f / num_iter, gamma_correction, true);
    }

    virtual CPT_CPU void render_online(
        int max_depth = 4
    ) override {
        CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
        size_t _num_bytes = 0;
        CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));

        accum_cnt ++;
        if (bidirectional) {
            render_pt_kernel<false><<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
                *camera, *verts, obj_info, aabbs, norms, uvs, 
                bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets,
                _cached_nodes, image, output_buffer, num_prims, num_objs, num_emitter, 
                accum_cnt * SEED_SCALER, max_depth, num_nodes, accum_cnt, num_cache
            ); 
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        }
        render_lt_kernel<true><<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
            *camera, *verts, obj_info, aabbs, norms, uvs, 
            bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets,
            _cached_nodes, image, output_buffer, num_prims, num_objs, num_emitter, 
            accum_cnt * SEED_SCALER, max_depth, num_nodes, 
            accum_cnt, num_cache, spec_constraint, caustic_scaling
        ); 
        CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
    }
};
