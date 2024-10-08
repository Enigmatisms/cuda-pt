/**
 * (W)ave(f)ront (S)imple path tracing with stream multiprocessing
 * We first consider how to make a WF tracer, then we start optimizing it, hence this is a 'Simple' one 
 * 
 * for each stream, we will create their own ray pools for
 * stream compaction and possible execution reordering
 * 
 * each stream contains 4 * 4 blocks, each block contains 16 * 16 threads, which is therefore
 * a 64 * 64 pixel patch. We will only create at most 8 streams, to fill up the host-device connections
 * therefore, it is recommended that the image sizes are the multiple of 64
 * 
 * for each kernel function, sx (int) and sy (int) are given, which is the base location of the current
 * stream. For example, let there be 4 streams and 4 kernel calls and the image is of size (256, 256)
 * stream 1: (0, 0), (64, 0), (128, 0), (192, 0)                |  1   2   3   4  |
 * stream 2: (0, 64), (64, 64), (128, 64), (192, 64)            |  1   2   3   4  |
 * stream 3: (0, 128), (64, 128), (128, 128), (192, 128)        |  1   2   3   4  |
 * stream 4: (0, 192), (64, 192), (128, 192), (192, 192)        |  1   2   3   4  |
 * 
 * @author Qianyue He
 * @date   2024.6.20
*/
#pragma once
#include <omp.h>
#include <cuda/pipeline>
#include <thrust/copy.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "renderer/wavefront_pt.cuh"

#include "core/progress.h"
#include "core/emitter.cuh"
#include "core/object.cuh"
#include "core/scene.cuh"
#include "core/stats.h"
#include "renderer/path_tracer.cuh"

// When doing profiling, this can be set as 1, otherwise, 8 is optimal
static constexpr int NUM_STREAM = 8;

class WavefrontPathTracer: public PathTracer {
private:
    using PathTracer::aabbs;
    using PathTracer::verts;
    using PathTracer::norms; 
    using PathTracer::uvs;
    using PathTracer::image;
    using PathTracer::num_prims;
    using PathTracer::w;
    using PathTracer::h;
    using PathTracer::obj_info;
    using PathTracer::num_objs;
    using PathTracer::num_emitter;
    using PathTracer::bvh_fronts;
    using PathTracer::bvh_backs;
    using PathTracer::node_fronts;
    using PathTracer::node_backs;
    using PathTracer::node_offsets;
    using PathTracer::camera;
public:
    /**
     * @param shapes    shape information (for ray intersection)
     * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
     * 
     * @todo: initialize emitters
     * @todo: initialize objects
    */
    WavefrontPathTracer(
        const Scene& scene,
        const PrecomputeAoS& _verts,
        const ArrayType<Vec3>& _norms, 
        const ArrayType<Vec2>& _uvs,
        int num_emitter
    ): PathTracer(scene, _verts, _norms, _uvs, num_emitter) {}
    
    CPT_CPU std::vector<uint8_t> render(
        int num_iter = 64,
        int max_depth = 4,
        bool gamma_correction = true
    ) override {
        TicToc _timer("render_pt_kernel()", num_iter);
        // step 1: create several streams (8 here)
        cudaStream_t streams[NUM_STREAM];

        const int x_patches = w / PATCH_X, y_patches = h / PATCH_Y;
        const int num_patches = x_patches * y_patches;
        PayLoadBufferSoA payload_buffer;
        payload_buffer.init(NUM_STREAM * PATCH_X, PATCH_Y);

        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamCreateWithFlags(&streams[i], cudaStreamDefault);

        // step 1, allocate 2D array of CUDA memory to hold: PathPayLoad
        thrust::device_vector<uint32_t> index_buffer(NUM_STREAM * TOTAL_RAY);
        uint32_t* const ray_idx_buffer = thrust::raw_pointer_cast(index_buffer.data());
        const dim3 GRID(BLOCK_X, BLOCK_Y), BLOCK(THREAD_X, THREAD_Y);

        for (int i = 0; i < num_iter; i++) {
            // here, we should use multi threading to submit the kernel call
            // each thread is responsible for only one stream (and dedicated to that stream only)
            // If we decide to use 8 streams, then we will use 8 CPU threads
            // Using multi-threading to submit kernel, we can avoid stucking on just one stream
            // This can be extended even further: use a high performance thread pool
            #pragma omp parallel for num_threads(NUM_STREAM)
            for (int p_idx = 0; p_idx < num_patches; p_idx++) {
                int patch_x = p_idx % x_patches, patch_y = p_idx / x_patches, stream_id = omp_get_thread_num();
                int stream_offset = stream_id * TOTAL_RAY;
                auto cur_stream = streams[stream_id];

                // step1: ray generator, generate the rays and store them in the PayLoadBuffer of a stream
                raygen_primary_hit_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                    *camera, *verts, payload_buffer, obj_info, aabbs, 
                    norms, uvs, bvh_fronts, bvh_backs, node_fronts, node_backs, 
                    node_offsets, ray_idx_buffer, stream_offset, num_prims, 
                    patch_x, patch_y, i, stream_id, image.w(), num_nodes);
                int num_valid_ray = TOTAL_RAY;
                auto start_iter = index_buffer.begin() + stream_id * TOTAL_RAY;
                for (int bounce = 0; bounce < max_depth; bounce ++) {
                   
                    // TODO: we can implement a RR shader here.
                    // step3: miss shader (ray inactive)
#ifndef FUSED_MISS_SHADER
                    miss_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        payload_buffer, ray_idx_buffer, stream_offset, num_valid_ray
                    );
#endif  // FUSED_MISS_SHADER
                    
                    // step4: thrust stream compaction (optional)
#ifndef NO_STREAM_COMPACTION
                    num_valid_ray = partition_func(
                        thrust::cuda::par.on(cur_stream), 
                        start_iter, start_iter + num_valid_ray,
                        ActiveRayFunctor(payload_buffer.ray_ds, NUM_STREAM * PATCH_X)
                    ) - start_iter;
#else
                    num_valid_ray = TOTAL_RAY;    
#endif  // NO_STREAM_COMPACTION

#ifndef NO_RAY_SORTING
                    // sort the ray (indices) by their ray tag (hit object)
                    // ray sorting is extremely slow
                    thrust::sort(
                        thrust::cuda::par.on(cur_stream), 
                        start_iter, start_iter + num_valid_ray,
                        RaySortFunctor(payload_buffer.ray_ds, NUM_STREAM * PATCH_X)
                    );
#endif

                    // here, if after partition, there is no valid PathPayLoad in the buffer, then we break from the for loop
                    if (!num_valid_ray) break;

                    // step5: NEE shader
                    nee_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        *verts, payload_buffer, obj_info, aabbs, norms, uvs, bvh_fronts, 
                        bvh_backs, node_fronts, node_backs, node_offsets, ray_idx_buffer, 
                        stream_offset, num_prims, num_objs, num_emitter, num_valid_ray, num_nodes
                    );

                    // step6: emission shader + ray update shader
                    bsdf_local_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        payload_buffer, obj_info, aabbs, uvs, ray_idx_buffer,
                        stream_offset, num_prims, num_valid_ray, bounce > 0
                    );

                    // step2: closesthit shader
                    if (bounce + 1 >= max_depth) break;
                    closesthit_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        *verts, payload_buffer, obj_info, aabbs, norms, uvs, 
                        bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets,
                        ray_idx_buffer, stream_offset, num_prims, num_valid_ray, num_nodes
                    );
                }

                // step8: accumulating radiance to the rgb buffer
                radiance_splat<<<GRID, BLOCK, 0, cur_stream>>>(
                    payload_buffer, image, stream_id, patch_x, patch_y
                );
            }

            // should we synchronize here? Yes, host end needs this
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
            printProgress(i, num_iter);
        }
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamDestroy(streams[i]);
        payload_buffer.destroy();
        printf("\n");

        // TODO: wavefront path tracing does not support online visualization yet
        return image.export_cpu(1.f / num_iter, gamma_correction);
    }
};
