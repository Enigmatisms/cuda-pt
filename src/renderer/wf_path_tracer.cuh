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
#include "core/progress.h"
#include "core/emitter.cuh"
#include "core/object.cuh"
#include "core/stats.h"
#include "renderer/path_tracer.cuh"

static constexpr int NUM_STREAM = 8;
#ifdef NO_STREAM_COMPACTION
    #define partition_func(...) (-1)
#else
    #ifdef STABLE_PARTITION
    #define partition_func(...) thrust::stable_partition(__VA_ARGS__)
    #else
    #define partition_func(...) thrust::partition(__VA_ARGS__)
    #endif    // STABLE_PARTITION
#endif    // NO_STREAM_COMPACTION

struct PathPayLoad {
    Vec4 thp;           // 4 * 4 Bytes
    Vec4 L;             // 4 * 4 Bytes

    Ray ray;            // 8 * 4 Bytes
    Sampler sg;         // 6 * 4 Bytes
    float pdf;          // 1 * 4 Byte:

    Interaction it;     // 5 * 4 Bytes

    CPT_CPU_GPU PathPayLoad(int sg_seed):
        thp(1, 1, 1, 1), L(0, 0, 0, 1),
        ray(Vec3(0, 0, 0), Vec3(0, 0, 1), 0), sg(sg_seed), pdf(1) {}

    // 28 * 4 Bytes in total
    CONDITION_TEMPLATE_2(T1, T2, Vec3)
    CPT_CPU_GPU PathPayLoad(T1&& o_, T2&& d_, int seed, float hitT = MAX_DIST, int offset = 0):
        thp(1, 1, 1, 1), L(0, 0, 0, 1),
        ray(std::forward<T1>(o_), std::forward<T2>(d_), hitT), sg(seed, offset) {}

    CPT_CPU_GPU PathPayLoad(float vthp, float vl, int seed, int offset = 0):
        thp(vthp, vthp, vthp), L(vl, vl, vl),
        ray(Vec3(0, 0, 0), Vec3(0, 0, 1), MAX_DIST), sg(seed, offset) {}

    CPT_CPU_GPU_INLINE void reset() {
        ray.clr_hit();
        ray.set_hit_index(0);
    }
};

namespace {
    using PayLoadBuffer = PathPayLoad* const;
    using PayLoadAccessor = Accessor<PathPayLoad>;
    using ConstPayLoadBuffer = const PayLoadBuffer;
}

// camera is defined in the global constant memory
// extern __constant__ DeviceCamera dev_cam;

/**
 * @brief ray generation kernel 
 * note that, all the kernels are called per stream, each stream can have multiple blocks (since it is a kernel call)
 * let's say, for example, a 4 * 4 block for one kernel call. These 16 blocks should be responsible for 
 * one image patch, offseted by the stream offset sx, sy.
 * @param (sx, sy) stream offset for the current image patch. For example,
 *      sx = row_patch_id * gridDim.x * blockDim.x
 *      sy = col_patch_id * gridDim.y * blockDim.y
 * 
 * @note we first consider images that have width and height to be the multiple of 128
 * to avoid having to consider the border problem
*/ 
CPT_KERNEL void raygen_shader(
    PayLoadBuffer payloads, 
    PayLoadAccessor acc, 
    uint32_t* const idx_buffer, 
    int sx, int sy, int pitch, int width
) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x + sx, py = threadIdx.y + blockIdx.y * blockDim.y + sy;
    const int block_index = (py - sy) * blockDim.x * gridDim.x + px - sx;
    auto& payload = acc(payloads, px, py, pitch);
    payload = PathPayLoad(px + py * width);
    payload.ray = dev_cam.generate_ray(px, py, payload.sg.next2D());
    // compress two int (to int16) to a uint32_t 
    // note that we can not use int here, since int shifting might retain the sign
    // it is implementation dependent
    idx_buffer[block_index] = (py << 16) + px;      
    __syncthreads();
}

/**
 * @brief find ray intersection for next hit pos
 * We first start with small pool size (4096), which can comprise at most 16 blocks
 * The ray pool is stream-compacted (with thrust::parition to remove the finished)
 * Note that we need an index buffer, since the Ray and Sampler are coupled
 * and we need the index to port the 
*/ 
CPT_KERNEL void closesthit_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    PayLoadBuffer payloads,
    PayLoadAccessor acc,
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    const uint32_t* const idx_buffer,
    int num_prims,
    int num_valid,
    int pitch
) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px, tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index], px = py & 0x0000ffff;
        py >>= 16;
        PathPayLoad payload = acc(payloads, px, py, pitch);         // To local register
        payload.reset();
        
        __shared__ Vec3 s_verts[TRI_IDX(BASE_ADDR)];                // vertex info
        __shared__ AABBWrapper s_aabbs[BASE_ADDR];                  // aabb

        ArrayType<Vec3> s_verts_arr(reinterpret_cast<Vec3*>(&s_verts[0]), BASE_ADDR);
        ShapeIntersectVisitor visitor(s_verts_arr, payload.ray, 0);
        ShapeExtractVisitor extract(*verts, *norms, *uvs, payload.ray, 0);

        Vec4 throughput(1, 1, 1), radiance(0, 0, 0);

        int num_copy = (num_prims + BASE_ADDR - 1) / BASE_ADDR, min_index = -1;   // round up
        payload.ray.hit_t = MAX_DIST;

        // ============= step 1: ray intersection =================
        #pragma unroll
        for (int cp_base = 0; cp_base < num_copy; ++cp_base) {
            // memory copy to shared memory
            int cur_idx = (cp_base << BASE_SHFL) + tid, remain_prims = min(num_prims - (cp_base << BASE_SHFL), BASE_ADDR);
            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

            // huge bug
            if (tid < BASE_ADDR && cur_idx < num_prims) {        // copy from gmem to smem
#ifdef USE_SOA
                cuda::memcpy_async(&s_verts[tid],                    &verts->x(cur_idx), sizeof(Vec3), pipe);
                cuda::memcpy_async(&s_verts[tid + BASE_ADDR],        &verts->y(cur_idx), sizeof(Vec3), pipe);
                cuda::memcpy_async(&s_verts[tid + (BASE_ADDR << 1)], &verts->z(cur_idx), sizeof(Vec3), pipe);
#else
                cuda::memcpy_async(&s_verts[TRI_IDX(tid)], &verts->data[TRI_IDX(cur_idx)], sizeof(Vec3) * 3, pipe);
#endif
                s_aabbs[tid].aabb.copy_from(aabbs[cur_idx]);
            }
            pipe.producer_commit();
            pipe.consumer_wait();
            __syncthreads();
            // this might not be a good solution
            payload.ray.hit_t = ray_intersect(payload.ray, shapes, s_aabbs, visitor, min_index, remain_prims, cp_base << BASE_SHFL, payload.ray.hit_t);
            __syncthreads();
        }

        // ============= step 2: local shading for indirect bounces ================
        if (min_index >= 0 && payload.thp.good()) {
            // if the ray hits nothing, or the path throughput is 0, then the ray will be inactive
            // inactive rays will only be processed in the miss_shader
            extract.set_index(min_index);
            payload.ray.set_hit();
            payload.ray.set_hit_index(min_index);
            payload.it = variant::apply_visitor(extract, shapes[min_index]);
        }

        auto& to_store = acc(payloads, px, py, pitch);
        to_store.ray = payload.ray;
        to_store.sg  = payload.sg;
        to_store.it  = payload.it;
    }
    __syncthreads();
}

/***
 * For non-delta hit (shading point), direct component should be evaluated:
 * we sample a light source then start ray intersection test
*/
CPT_KERNEL void nee_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstShapePtr shapes,
    ConstAABBPtr aabbs,
    ConstPrimPtr verts,
    PayLoadBuffer payloads,
    PayLoadAccessor acc,
    ConstPrimPtr norms, 
    ConstUVPtr,         
    const int* const idx_buffer,
    int num_prims,
    int num_objects,
    int num_emitter,
    int num_valid,
    int pitch
) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px, tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index], px = py & 0x0000ffff;
        py >>= 16;
        PathPayLoad payload = acc(payloads, px, py, pitch);         // To local register

        int object_id   = prim2obj[payload.ray.hit_id()],
            material_id = objects[object_id].bsdf_id,
            emitter_id  = objects[object_id].emitter_id;
        bool hit_emitter = emitter_id > 0;

        float direct_pdf = 1;

        Emitter* emitter = sample_emitter(payload.sg, direct_pdf, num_emitter, emitter_id);
        int emitter_id = objects[emitter->get_obj_ref()].sample_emitter_primitive(payload.sg.discrete1D(), direct_pdf);
        Ray shadow_ray(payload.ray.advance(payload.ray.hit_t), Vec3(0, 0, 0));
        // use ray.o to avoid creating another shadow_int variable
        Vec4 direct_comp(0, 0, 0, 1);
        shadow_ray.d = emitter->sample(shadow_ray.o, direct_comp, direct_pdf, payload.sg.next2D(), verts, norms, emitter_id) - shadow_ray.o;

        float emit_len_mis = shadow_ray.d.length();
        shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized directi 
        // (3) NEE scene intersection test (possible warp divergence, but... nevermind)
        if (emitter != c_emitter[0] && occlusion_test(shadow_ray, objects, shapes, aabbs, *verts, num_objects, emit_len_mis - EPSILON)) {
            // MIS for BSDF / light sampling, to achieve better rendering
            // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
            emit_len_mis = direct_pdf + c_material[material_id]->pdf(payload.it, shadow_ray.d, payload.ray.d) * emitter->non_delta();
            payload.L += payload.thp * direct_comp * c_material[material_id]->eval(payload.it, shadow_ray.d, payload.ray.d) * \
                (float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
            // numerical guard, in case emit_len_mis is 0
        }

        auto& to_store = acc(payloads, px, py, pitch);
        to_store.ray = payload.ray;
        to_store.sg  = payload.sg;
        to_store.thp = payload.thp;
    }
    __syncthreads();
}


/**
 * BSDF sampling & direct shading shader
*/

CPT_KERNEL void bsdf_emission_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    PayLoadBuffer payloads,
    PayLoadAccessor acc,
    ConstUVPtr,         
    const int* const idx_buffer,
    int num_prims, 
    int num_valid,
    int pitch
    bool secondary_bounce,
) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px;

    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index], px = py & 0x0000ffff;
        py >>= 16;
        PathPayLoad payload = acc(payloads, px, py, pitch);         // To local register

        int object_id   = prim2obj[payload.ray.hit_id()],
            material_id = objects[object_id].bsdf_id,
            emitter_id  = objects[object_id].emitter_id;
        bool hit_emitter = emitter_id > 0;

        // emitter MIS
        float emission_weight = payload.pdf / (payload.pdf + 
                objects[object_id].solid_angle_pdf(payload.it.shading_norm, payload.ray.d, payload.ray.hit_t) * hit_emitter * secondary_bounce);
        // (2) check if the ray hits an emitter
        Vec4 direct_comp = payload.thp *\
                    c_emitter[emitter_id]->eval_le(&payload.ray.d, &payload.it.shading_norm);
        payload.L += direct_comp * emission_weight;

        payload.ray.o = payload.ray.advance(payload.ray.hit_t);
        payload.ray.d = c_material[material_id]->sample_dir(payload.ray.d, payload.it, payload.thp, emission_weight, payload.sg.next2D());

        auto& to_store = acc(payloads, px, py, pitch);
        to_store.ray = payload.ray;
        to_store.thp = payload.thp;
        to_store.sg  = payload.sg;
        to_store.L   = payload.L;
    }
    __syncthreads();
}

/**
 * Sample the new ray according to BSDF
*/
CPT_KERNEL void ray_update_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    PayLoadBuffer payloads,
    PayLoadAccessor acc,
    ConstUVPtr,   
    const int* const idx_buffer,
    int num_valid,
    int pitch
) {
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px;

    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index], px = py & 0x0000ffff;
        py >>= 16;
        PathPayLoad payload = acc(payloads, px, py, pitch);         // To local register

        int object_id   = prim2obj[payload.ray.hit_id()],
            material_id = objects[object_id].bsdf_id,
            emitter_id  = objects[object_id].emitter_id;
        
        PathPayLoad payload = payloads[block_index];
        payload.ray.o = payload.ray.advance(payload.ray.hit_t);
        payload.ray.d = c_material[material_id]->sample_dir(
            payload.ray.d, payload.it, payload.thp, payload.pdf, payload.sg.next2D()
        );
        auto& to_store = acc(payloads, px, py, pitch);
        to_store.ray = payload.ray;
    }
    __syncthreads();
}

/**
 * Purpose of the miss shader: if ray hits nothing in closesthit shader
 * the we will set the hit status (flag) to be false
 * in this shader, we find the rays marked as no-hit, and check the
 * availability of environment map (currently not supported)
 * after processing the env-map lighting, we mark the ray as inactive
 * before stream compaction. Then stream compaction will 'remove' all these
 * rays (and the threads)
 * 
 * MISS_SHADER is the only place where you mark a ray as inactive
*/
CPT_KERNEL void miss_shader(
    PayLoadBuffer payloads,
    PayLoadAccessor acc,
    const int* const idx_buffer,
    int num_valid,
    int pitch
) {
    // Nothing here, currently, if we decide not to support env lighting

    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px;

    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index], px = py & 0x0000ffff;
        py >>= 16;
        PathPayLoad payload = acc(payloads, px, py, pitch);         // To local register
        if (!payload.ray.is_hit()) {
            // TODO: process no-hit ray, environment map lighting
            payload.ray.set_active(false);
        }
        auto& to_store = acc(payloads, px, py, pitch);
        to_store.ray = payload.ray;
    }
    __syncthreads();
}

CPT_KERNEL void radiance_splat(
    PayLoadBuffer payloads,
    PayLoadAccessor acc,
    DeviceImage image,
    int pitch
) {
    // Nothing here, currently, if we decide not to support env lighting
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    const int block_index = py * blockDim.x * gridDim.x + px;
    Vec4 payload = acc(payloads, px, py, pitch).L;         // To local register
    image(px, py) += L.numeric_err() ? Vec4(0, 0, 0, 1) : L;
    __syncthreads();
}

/**
 * This functor is used for stream compaction
*/
struct ActiveRayFunctor
{
    const PayLoadBuffer data;
    const PayLoadAccessor acco;
    const int pitch;

    CPT_CPU_GPU ActiveRayFunctor(PayLoadBuffer _data, PayLoadAccessor&& _acc, size_t pit):
        data(_data), _acc(std::move(acco)), pit(pitch) {}

    CPT_CPU_GPU bool operator()(int index) const
    {
        uint32_t py = index >> 16, px = index & 0x0000ffff;
        return acco(data, px, py, pitch).ray.is_active();
    }
};

class WavefrontPathTracer: public TracerBase {
using TracerBase::shapes;
using TracerBase::aabbs;
using TracerBase::verts;
using TracerBase::norms; 
using TracerBase::uvs;
using TracerBase::image;
using TracerBase::dev_image;
using TracerBase::num_prims;
using TracerBase::w;
using TracerBase::h;
private:
    ObjInfo* obj_info;
    int*    prim2obj;
    int num_objs;
    int num_emitter;
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
        const std::vector<ObjInfo>& _objs,
        const std::vector<Shape>& _shapes,
        const ArrayType<Vec3>& _verts,
        const ArrayType<Vec3>& _norms, 
        const ArrayType<Vec2>& _uvs,
        int num_emitter,
        int width, int height
    ): TracerBase(_shapes, _verts, _norms, _uvs, width, height), 
        num_objs(_objs.size()), num_emitter(num_emitter) 
    {
        CUDA_CHECK_RETURN(cudaMallocManaged(&obj_info, num_objs * sizeof(ObjInfo)));
        CUDA_CHECK_RETURN(cudaMallocManaged(&prim2obj, num_prims * sizeof(int)));

        int prim_offset = 0;
        for (int i = 0; i < num_objs; i++) {
            obj_info[i] = _objs[i];
            int prim_num = _objs[i].prim_num;
            for (int j = 0; j < prim_num; j++)
                prim2obj[prim_offset + j] = i;
            prim_offset += prim_num;
        }
        // TODO: copy all the material into constant memory
    }

    ~WavefrontPathTracer() {
        CUDA_CHECK_RETURN(cudaFree(obj_info));
        CUDA_CHECK_RETURN(cudaFree(prim2obj));
    }

    CPT_CPU std::vector<uint8_t> render(
        int num_iter = 64,
        int max_depth = 4,
        bool gamma_correction = true
    ) override {
        TicToc _timer("render_pt_kernel()", num_iter);
        // step 1: create several streams (8 here)
        cudaStream_t streams[NUM_STREAM];

        // optimize this
        constexpr int BLOCK_X = 4;
        constexpr int BLOCK_Y = 4;
        constexpr int THREAD_X = 16;
        constexpr int THREAD_Y = 16;
        constexpr int PATCH_X = BLOCK_X * THREAD_X;
        constexpr int PATCH_Y = BLOCK_Y * THREAD_Y;
        constexpr int TOTAL_RAY = PATCH_X * PATCH_Y;
        const int x_patches = w / PATCH_X, y_patches = h / PATCH_Y;
        const int num_patches = x_patches * y_patches;

        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamCreateWithFlags(&streams[i], cudaStreamDefault);


        assert(omp_get_num_threads() == NUM_STREAM);

        // step 1, allocate 2D array of CUDA memory to hold: PathPayLoad
        uint32_t* ray_idx_buffer;
        DevicePitchBuffer<PathPayLoad> payload_buffer(NUM_STREAM * PATCH_X, PATCH_Y);
        CUDA_CHECK_RETURN(cudaMalloc(&ray_idx_buffer, sizeof(uint32_t) * NUM_STREAM * TOTAL_RAY));
        const dim3 GRID(BLOCK_X, BLOCK_Y), BLOCK(THREAD_X, THREAD_Y);
        const int pitch = payload_buffer.get_pitch();

        for (int i = 0; i < num_iter; i++) {
            // here, we should use multi threading to submit the kernel call
            // each thread is responsible for only one stream (and dedicated to that stream only)
            // If we decide to use 8 streams, then we will use 8 CPU threads
            // Using multi-threading to submit kernel, we can avoid stucking on just one stream
            // This can be extended even further: use a high performance thread pool
            #pragma omp parallel for num_threads(NUM_STREAM)
            for (int p_idx = 0; p_idx < num_patches; p_idx) {
                int patch_x = p_idx % x_patches, patch_y = p_idx / x_patches, stream_id = omp_get_thread_num();
                int sx      = patch_x * PATCH_X, sy      = patch_y * PATCH_Y;
                auto cur_stream = streams[stream_id];

                // step1: ray generator, generate the rays and store them in the PayLoadBuffer of a stream
                raygen_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                    payload_buffer.data(), payload_buffer.accessor(), 
                    ray_idx_buffer, sx, sy, pitch, image.w());
                int num_valid_ray = TOTAL_RAY;
                for (int bounce = 0; bounce < max_depth; bounce ++) {
                    // step2: closesthit shader
                    closesthit_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        obj_info, prim2obj, shapes, aabbs, verts, payload_buffer.data(),
                        payload_buffer.accessor(), norms, uvs, ray_idx_buffer,
                        num_prims, num_valid_ray, pitch
                    );
                    
                    // step3: thrust stream compaction (optional)
                    num_valid_ray = partition_func(
                        thrust::cuda::par.on(cur_stream), 
                        &ray_idx_buffer[stream_id * TOTAL_RAY], 
                        &ray_idx_buffer[stream_id * TOTAL_RAY + TOTAL_RAY],
                        ActiveRayFunctor(payload_buffer.data(), payload_buffer.accessor(), pitch)
                    );
                    // here, if after partition, there is no valid PathPayLoad in the buffer, then we break from the for loop
                    if (!num_valid_ray) break;
                    num_valid_ray = num_valid_ray < 0 ? TOTAL_RAY : num_valid_ray;

                    // step4: nee shader
                    nee_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        obj_info, prim2obj, shapes, aabbs, verts, payload_buffer.data(),
                        payload_buffer.accessor(), norms, uvs, ray_idx_buffer,
                        num_prims, num_objects, num_emitter, num_valid_ray, pitch
                    );

                    // step5: emission shader
                    bsdf_emission_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        obj_info, prim2obj, payload_buffer.data(),
                        payload_buffer.accessor(), uvs, ray_idx_buffer,
                        num_prims, num_valid_ray, pitch, bounce > 0
                    );

                    // step6: rayupdate shader
                    ray_update_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        obj_info, prim2obj, payload_buffer.data(),
                        payload_buffer.accessor(), uvs, ray_idx_buffer,
                        num_valid_ray, pitch
                    );
                }

                // step7: accumulating radiance to the rgb buffer
                radiance_splat<<<GRID, BLOCK, 0, cur_stream>>>(
                    payload_buffer.data(), payload_buffer.accessor(), dev_image, pitch
                );
            }

            // should we synchronize here? Yes, host end needs this
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
            printProgress(i, num_iter);
        }
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamDestroy(streams[i]);
        CUDA_CHECK_RETURN(cudaFree(ray_idx_buffer));
        printf("\n");
        return image.export_cpu(1.f / num_iter, gamma_correction);
    }
};
