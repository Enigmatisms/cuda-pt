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
//#define NO_STREAM_COMPACTION
#define STABLE_PARTITION
#ifdef STABLE_PARTITION
#define partition_func(...) thrust::stable_partition(__VA_ARGS__)
#else
#define partition_func(...) thrust::partition(__VA_ARGS__)
#endif    // STABLE_PARTITION

union PDFInteraction {
    struct {
        float pdf;
        Interaction it;
    } v;
    struct {
        uint4 p1;
        uint4 p2;
        uint4 p3;
    } data;

    CPT_CPU_GPU PDFInteraction() {}
    CPT_CPU_GPU_INLINE PDFInteraction(float pdf) { v.pdf = pdf; }

    CPT_CPU_GPU_INLINE Interaction& it() { return this->v.it; }    
    CPT_CPU_GPU_INLINE const Interaction& it_const() const { return this->v.it; }    

    CPT_CPU_GPU_INLINE float& pdf() { return this->v.pdf; }    
    CPT_CPU_GPU_INLINE float pdf_v() const { return this->v.pdf; }    

};

class PayLoadBufferSoA {
public:
    Vec4* thps;             // direct malloc
    Vec4* Ls;               // direct malloc
    
    Ray* rays;              // direct malloc
    Sampler* samplers;      // pitch malloc
    PDFInteraction* its;    // its

    size_t sp_pitch;
    size_t it_pitch;
    int    _width;       

    CPT_CPU_GPU PayLoadBufferSoA() {}

    CPT_CPU void init(int width, int height) {
        _width = width;
        int full_size = width * height;
        CUDA_CHECK_RETURN(cudaMalloc(&thps, sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&Ls,   sizeof(Vec4) * full_size));
        CUDA_CHECK_RETURN(cudaMalloc(&rays, sizeof(Ray) * full_size));
        CUDA_CHECK_RETURN(cudaMallocPitch(&samplers, &sp_pitch, sizeof(Sampler) * width, height));
        CUDA_CHECK_RETURN(cudaMallocPitch(&its,      &it_pitch, sizeof(PDFInteraction) * width, height));
    }

    CPT_CPU_GPU_INLINE Vec4& thp(int col, int row) { return thps[col + row * _width]; }
    CPT_CPU_GPU_INLINE const Vec4& thp(int col, int row) const { return thps[col + row * _width]; }

    CPT_CPU_GPU_INLINE Vec4& L(int col, int row) { return Ls[col + row * _width]; }
    CPT_CPU_GPU_INLINE const Vec4& L(int col, int row) const { return Ls[col + row * _width]; }

    CPT_CPU_GPU_INLINE Ray& ray(int col, int row) { return rays[col + row * _width]; }
    CPT_CPU_GPU_INLINE const Ray& ray(int col, int row) const { return rays[col + row * _width]; }

    CPT_CPU_GPU_INLINE Sampler& sampler(int col, int row) 
        { return ((Sampler*)((char*)samplers + row * sp_pitch))[col]; }
    CPT_CPU_GPU_INLINE const Sampler& sampler(int col, int row) const 
        { return ((Sampler*)((char*)samplers + row * sp_pitch))[col]; }

    CPT_CPU_GPU_INLINE PDFInteraction& iteraction(int col, int row) 
        { return ((PDFInteraction*)((char*)its + row * it_pitch))[col]; }
    CPT_CPU_GPU_INLINE const PDFInteraction& iteraction(int col, int row) const 
        { return ((PDFInteraction*)((char*)its + row * it_pitch))[col]; }

    CPT_CPU void destroy() {
        CUDA_CHECK_RETURN(cudaFree(thps));
        CUDA_CHECK_RETURN(cudaFree(Ls));
        CUDA_CHECK_RETURN(cudaFree(rays));
        CUDA_CHECK_RETURN(cudaFree(samplers));
        CUDA_CHECK_RETURN(cudaFree(its));
    }
};  

extern __constant__ PayLoadBufferSoA payloads;

namespace {
    using PayLoadBuffer      = PayLoadBufferSoA* const;
    using ConstPayLoadBuffer = const PayLoadBuffer;
    using IndexBuffer        = uint32_t* const;

    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;
    constexpr int THREAD_X = 16;
    constexpr int THREAD_Y = 16;
    constexpr int PATCH_X = BLOCK_X * THREAD_X;
    constexpr int PATCH_Y = BLOCK_Y * THREAD_Y;
    constexpr int TOTAL_RAY = PATCH_X * PATCH_Y;
}

// camera is defined in the global constant memory
// extern __constant__ DeviceCamera dev_cam;

/**
 * @brief ray generation kernel 
 * note that, all the kernels are called per stream, each stream can have multiple blocks (since it is a kernel call)
 * let's say, for example, a 4 * 4 block for one kernel call. These 16 blocks should be responsible for 
 * one image patch, offseted by the stream_offset.
 * @note we first consider images that have width and height to be the multiple of 128
 * to avoid having to consider the border problem
*/ 
CPT_KERNEL void raygen_shader(
    IndexBuffer idx_buffer, 
    int x_patch, int y_patch, int iter,
    int stream_id, int width
) {
    // stream and patch related offset
    const int sx = x_patch * PATCH_X, sy = y_patch * PATCH_Y, buffer_xoffset = stream_id * PATCH_X;
    // within kernel thread position
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    // linear idx_buffer position
    const int block_index = py * blockDim.x * gridDim.x + px;

    Sampler sg = Sampler(px + sx + (py + sy) * width, iter * SEED_SCALER);
    // Hopefully, this will give us more coalesced memory access
    payloads.ray(px + buffer_xoffset, py) =
            dev_cam.generate_ray(px + sx, py + sy, sg.next2D());
    payloads.thp(px + buffer_xoffset, py) = Vec4(1, 1, 1, 1);
    payloads.L(px + buffer_xoffset, py)   = Vec4(0, 0, 0, 1);
    payloads.sampler(px + buffer_xoffset, py)    = sg;
    payloads.iteraction(px + buffer_xoffset, py) = PDFInteraction(1.f);

    // compress two int (to int16) to a uint32_t 
    // note that we can not use int here, since int shifting might retain the sign
    // it is implementation dependent
    // note that we only have stream_number * payloadbuffers
    // so row indices won't be offset by sy, col indices should only be offseted by stream_offset
    idx_buffer[block_index + stream_id * TOTAL_RAY] = (py << 16) + px + buffer_xoffset;      
    // px has already encoded stream_offset (stream_id * PATCH_X)
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
    ConstPrimPtr norms, 
    ConstUVPtr uvs,
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims,
    int num_valid
) {
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x,              // px
                            tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
    py >>= 16;
    Ray           ray = payloads.ray(px, py);
    PDFInteraction it = payloads.iteraction(px, py);            // To local register
    ray.reset();
    
    __shared__ Vec3 s_verts[TRI_IDX(BASE_ADDR)];                // vertex info
    __shared__ AABBWrapper s_aabbs[BASE_ADDR];                  // aabb

    ArrayType<Vec3> s_verts_arr(reinterpret_cast<Vec3*>(&s_verts[0]), BASE_ADDR);
    ShapeIntersectVisitor visitor(s_verts_arr, ray, 0);
    ShapeExtractVisitor extract(*verts, *norms, *uvs, ray, 0);

    int num_copy = (num_prims + BASE_ADDR - 1) / BASE_ADDR, min_index = -1;   // round up
    ray.hit_t = MAX_DIST;

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
        ray.hit_t = ray_intersect(ray, shapes, s_aabbs, visitor, min_index, remain_prims, cp_base << BASE_SHFL, ray.hit_t);
        __syncthreads();
    }

    // ============= step 2: local shading for indirect bounces ================
    if (block_index < num_valid && min_index >= 0) {
        // if the ray hits nothing, or the path throughput is 0, then the ray will be inactive
        // inactive rays will only be processed in the miss_shader
        extract.set_index(min_index);
        ray.set_hit();
        ray.set_hit_index(min_index);
        it.it() = variant::apply_visitor(extract, shapes[min_index]);
    }

    payloads.ray(px, py)        = ray;
    payloads.iteraction(px, py) = it;
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
    ConstPrimPtr norms, 
    ConstUVPtr,         
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims,
    int num_objects,
    int num_emitter,
    int num_valid
) {
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px
    
    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
        py >>= 16;
        Vec4 thp = payloads.thp(px, py);
        Ray ray  = payloads.ray(px, py);
        Sampler sg = payloads.sampler(px, py);
        const PDFInteraction it = payloads.iteraction(px, py);

        int object_id    = prim2obj[ray.hit_id()],
            material_id  = objects[object_id].bsdf_id,
            emitter_id   = objects[object_id].emitter_id;

        float direct_pdf = 1;

        Emitter* emitter = sample_emitter(sg, direct_pdf, num_emitter, emitter_id);
        emitter_id       = objects[emitter->get_obj_ref()].sample_emitter_primitive(sg.discrete1D(), direct_pdf);
        Ray shadow_ray(ray.advance(ray.hit_t), Vec3(0, 0, 0));
        // use ray.o to avoid creating another shadow_int variable
        Vec4 direct_comp(0, 0, 0, 1);
        shadow_ray.d = emitter->sample(shadow_ray.o, direct_comp, direct_pdf, sg.next2D(), verts, norms, emitter_id) - shadow_ray.o;

        float emit_len_mis = shadow_ray.d.length();
        shadow_ray.d *= __frcp_rn(emit_len_mis);              // normalized direct
        // (3) NEE scene intersection test (possible warp divergence, but... nevermind)
        if (emitter != c_emitter[0] && occlusion_test(shadow_ray, objects, shapes, aabbs, *verts, num_objects, emit_len_mis - EPSILON)) {
            // MIS for BSDF / light sampling, to achieve better rendering
            // 1 / (direct + ...) is mis_weight direct_pdf / (direct_pdf + material_pdf), divided by direct_pdf
            emit_len_mis = direct_pdf + c_material[material_id]->pdf(it.it_const(), shadow_ray.d, ray.d) * emitter->non_delta();
            payloads.L(px, py) += thp * direct_comp * c_material[material_id]->eval(it.it_const(), shadow_ray.d, ray.d) * \
                (float(emit_len_mis > EPSILON) * __frcp_rn(emit_len_mis < EPSILON ? 1.f : emit_len_mis));
            // numerical guard, in case emit_len_mis is 0
        }

        payloads.sampler(px, py) = sg;
    }
    __syncthreads();
}


/**
 * BSDF sampling & direct shading shader
*/

CPT_KERNEL void bsdf_emission_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstUVPtr,         
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_prims, 
    int num_valid,
    bool secondary_bounce
) {
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px

    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
        py >>= 16;

        Vec4 thp = payloads.thp(px, py);
        Ray ray  = payloads.ray(px, py);
        PDFInteraction it = payloads.iteraction(px, py);

        int object_id   = prim2obj[ray.hit_id()],
            emitter_id  = objects[object_id].emitter_id;
        bool hit_emitter = emitter_id > 0;

        // emitter MIS
        float emission_weight = it.pdf_v() / (it.pdf_v() + 
                objects[object_id].solid_angle_pdf(it.it_const().shading_norm, ray.d, ray.hit_t) * hit_emitter * secondary_bounce);
        // (2) check if the ray hits an emitter
        Vec4 direct_comp = thp *\
                    c_emitter[emitter_id]->eval_le(&ray.d, &it.it_const().shading_norm);

        payloads.L(px, py) += direct_comp * emission_weight;
    }
    __syncthreads();
}

/**
 * Sample the new ray according to BSDF
*/
CPT_KERNEL void ray_update_shader(
    ConstObjPtr objects,
    ConstIndexPtr prim2obj,
    ConstUVPtr,   
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_valid
) {
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px

    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
        py >>= 16;

        Vec4 thp = payloads.thp(px, py);
        Ray ray  = payloads.ray(px, py);
        Sampler sg = payloads.sampler(px, py);
        PDFInteraction it = payloads.iteraction(px, py);

        int object_id   = prim2obj[ray.hit_id()],
            material_id = objects[object_id].bsdf_id;
        
        ray.o = ray.advance(ray.hit_t);
        ray.d = c_material[material_id]->sample_dir(
            ray.d, it.it_const(), thp, it.pdf(), sg.next2D()
        );

        payloads.thp(px, py) = thp;
        payloads.ray(px, py) = ray;
        payloads.sampler(px, py) = sg;
        payloads.iteraction(px, py) = it;
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
    const IndexBuffer idx_buffer,
    int stream_offset,
    int num_valid
) {
    // Nothing here, currently, if we decide not to support env lighting
    const int block_index = (threadIdx.y + blockIdx.y * blockDim.y) *           // py
                            blockDim.x * gridDim.x +                            // cols
                            threadIdx.x + blockIdx.x * blockDim.x;              // px
    if (block_index < num_valid) {
        uint32_t py = idx_buffer[block_index + stream_offset], px = py & 0x0000ffff;
        py >>= 16;
        Vec4 thp = payloads.thp(px, py);
        Ray& ray = payloads.ray(px, py);         // To local register
        if ((!ray.is_hit()) || thp.max_elem() <= 1e-5f) {
            // TODO: process no-hit ray, environment map lighting
            ray.set_active(false);
        }
    }
    __syncthreads();
}

CPT_KERNEL void radiance_splat(
    DeviceImage& image, int stream_id, int x_patch, int y_patch
) {
    // Nothing here, currently, if we decide not to support env lighting
    const int px = threadIdx.x + blockIdx.x * blockDim.x, py = threadIdx.y + blockIdx.y * blockDim.y;
    Vec4 L = payloads.L(px + stream_id * PATCH_X, py);         // To local register
    image(px + x_patch * PATCH_X, py + y_patch * PATCH_Y) += L.numeric_err() ? Vec4(0, 0, 0, 1) : L;
    __syncthreads();
}

/**
 * This functor is used for stream compaction
*/
struct ActiveRayFunctor
{
    const int width;

    CPT_CPU_GPU ActiveRayFunctor(int w): width(w) {}

    CPT_GPU_INLINE bool operator()(uint32_t index) const
    {
        uint32_t py = index >> 16, px = index & 0x0000ffff;
        return payloads.rays[px + py * width].is_active();
    }
};

class WavefrontPathTracer: public PathTracer {
private:
    using PathTracer::shapes;
    using PathTracer::aabbs;
    using PathTracer::verts;
    using PathTracer::norms; 
    using PathTracer::uvs;
    using PathTracer::image;
    using PathTracer::dev_image;
    using PathTracer::num_prims;
    using PathTracer::w;
    using PathTracer::h;
    using PathTracer::obj_info;
    using PathTracer::prim2obj;
    using PathTracer::num_objs;
    using PathTracer::num_emitter;
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
    ): PathTracer(_objs, _shapes, _verts, _norms, _uvs, num_emitter, width, height) {}
    
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
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(payloads, &payload_buffer, sizeof(PayLoadBufferSoA)));

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
                raygen_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                    ray_idx_buffer, patch_x, patch_y, i, stream_id, image.w());
                int num_valid_ray = TOTAL_RAY;
                auto start_iter = index_buffer.begin() + stream_id * TOTAL_RAY;
                for (int bounce = 0; bounce < max_depth; bounce ++) {
                    // step2: closesthit shader
                    closesthit_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        obj_info, prim2obj, shapes, aabbs, verts,
                        norms, uvs, ray_idx_buffer,
                        stream_offset, num_prims, num_valid_ray
                    );

                    // step3: miss shader (ray inactive)
                    miss_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        ray_idx_buffer, stream_offset, num_valid_ray
                    );
                    
                    // step4: thrust stream compaction (optional)
#ifndef NO_STREAM_COMPACTION
                    num_valid_ray = partition_func(
                        thrust::cuda::par.on(cur_stream), 
                        start_iter, start_iter + num_valid_ray,
                        ActiveRayFunctor(NUM_STREAM * PATCH_X)
                    ) - start_iter;
#else
                    num_valid_ray = TOTAL_RAY;    
#endif  // NO_STREAM_COMPACTION

                    // here, if after partition, there is no valid PathPayLoad in the buffer, then we break from the for loop
                    if (!num_valid_ray) break;

                    // step5: NEE shader
                    nee_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        obj_info, prim2obj, shapes, aabbs, verts, norms, uvs, ray_idx_buffer,
                        stream_offset, num_prims, num_objs, num_emitter, num_valid_ray
                    );

                    // step6: emission shader
                    bsdf_emission_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        obj_info, prim2obj, uvs, ray_idx_buffer,
                        stream_offset, num_prims, num_valid_ray, bounce > 0
                    );

                    // step7: rayupdate shader
                    ray_update_shader<<<GRID, BLOCK, 0, cur_stream>>>(
                        obj_info, prim2obj, uvs, ray_idx_buffer,
                        stream_offset, num_valid_ray
                    );
                }

                // step8: accumulating radiance to the rgb buffer
                radiance_splat<<<GRID, BLOCK, 0, cur_stream>>>(
                    *dev_image, stream_id, patch_x, patch_y
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
        return image.export_cpu(1.f / num_iter, gamma_correction);
    }
};
