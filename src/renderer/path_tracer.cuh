/**
 * Simple tile-based path tracer
 * @date: 5.12.2024
 * @author: Qianyue He
*/
#pragma once
#include <cuda/pipeline>
#include <cuda_gl_interop.h>
#include "core/stats.h"
#include "core/scene.cuh"
#include "core/progress.h"
#include "renderer/tracer_base.cuh"
#include "renderer/megakernel_pt.cuh"

static constexpr int SEED_SCALER = 11451;

class PathTracer: public TracerBase {
private:
    float4* _bvh_fronts;
    float4* _bvh_backs;
    float4* _node_fronts;
    float4* _node_backs;
    int* _node_offsets;
protected:
    using TracerBase::aabbs;
    using TracerBase::verts;
    using TracerBase::norms; 
    using TracerBase::uvs;
    using TracerBase::image;
    using TracerBase::num_prims;
    using TracerBase::w;
    using TracerBase::h;

    ObjInfo* obj_info;
    int num_objs;
    int num_nodes;
    int num_emitter;

    cudaTextureObject_t bvh_fronts;
    cudaTextureObject_t bvh_backs;
    cudaTextureObject_t node_fronts;
    cudaTextureObject_t node_backs;
    cudaTextureObject_t node_offsets;

    DeviceCamera* camera;

    // IMGUI related 
    uint32_t cuda_texture_id, pbo_id;
    cudaGraphicsResource_t pbo_resc;
    float* output_buffer;                // output buffer for images
    int accum_cnt;
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
    PathTracer(
        const Scene& scene,
        const PrecomputeAoS& _verts,
        const ArrayType<Vec3>& _norms, 
        const ArrayType<Vec2>& _uvs,
        int num_emitter
    ): TracerBase(scene.shapes, _verts, _norms, _uvs, scene.config.width, scene.config.height), 
        num_objs(scene.objects.size()), num_nodes(-1), num_emitter(num_emitter), 
        cuda_texture_id(0), pbo_id(0), output_buffer(nullptr), accum_cnt(0)
    {
        // TODO: export BVH here, if the scene BVH is available
#ifdef RENDERER_USE_BVH
        if (scene.bvh_available()) {
            size_t num_bvh  = scene.bvh_fronts.size();
            num_nodes = scene.node_fronts.size();
            CUDA_CHECK_RETURN(cudaMalloc(&_bvh_fronts,  num_bvh * sizeof(float4)));
            CUDA_CHECK_RETURN(cudaMalloc(&_bvh_backs,   num_bvh * sizeof(float4)));
            CUDA_CHECK_RETURN(cudaMalloc(&_node_fronts, num_nodes * sizeof(float4)));
            CUDA_CHECK_RETURN(cudaMalloc(&_node_backs,  num_nodes * sizeof(float4)));
            CUDA_CHECK_RETURN(cudaMalloc(& _node_offsets, num_nodes * sizeof(int)));
            PathTracer::createTexture1D<float4>(scene.bvh_fronts.data(),  num_bvh,   _bvh_fronts,  bvh_fronts);
            PathTracer::createTexture1D<float4>(scene.bvh_backs.data(),   num_bvh,   _bvh_backs,   bvh_backs);
            PathTracer::createTexture1D<float4>(scene.node_fronts.data(), num_nodes, _node_fronts, node_fronts);
            PathTracer::createTexture1D<float4>(scene.node_backs.data(),  num_nodes, _node_backs,  node_backs);
            PathTracer::createTexture1D<int>(scene.node_offsets.data(), num_nodes, _node_offsets, node_offsets);
        } else {
            throw std::runtime_error("BVH not available in scene. Abort.");
        }
#endif  // RENDERER_USE_BVH

        CUDA_CHECK_RETURN(cudaMallocManaged(&obj_info, num_objs * sizeof(ObjInfo)));
        CUDA_CHECK_RETURN(cudaMalloc(&camera, sizeof(DeviceCamera)));
        CUDA_CHECK_RETURN(cudaMemcpy(camera, &scene.cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice));

        for (int i = 0; i < num_objs; i++)
            obj_info[i] = scene.objects[i];
    }

    virtual ~PathTracer() {
        CUDA_CHECK_RETURN(cudaFree(obj_info));
        CUDA_CHECK_RETURN(cudaFree(camera));
#ifdef RENDERER_USE_BVH
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(bvh_fronts));
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(bvh_backs));
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(node_fronts));
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(node_backs));
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(node_offsets));
        CUDA_CHECK_RETURN(cudaFree(_bvh_fronts));
        CUDA_CHECK_RETURN(cudaFree(_bvh_backs));
        CUDA_CHECK_RETURN(cudaFree(_node_fronts));
        CUDA_CHECK_RETURN(cudaFree(_node_backs));
        CUDA_CHECK_RETURN(cudaFree(_node_offsets));
#endif  // RENDERER_USE_BVH
    }

    CPT_CPU uint32_t& get_texture_id() noexcept { return this->cuda_texture_id; }
    CPT_CPU uint32_t& get_pbo_id()     noexcept { return this->pbo_id; }

    virtual CPT_CPU std::vector<uint8_t> render(
        int num_iter = 64,
        int max_depth = 4,
        bool gamma_correction = true
    ) override {
        printf("Rendering starts.\n");
        TicToc _timer("render_pt_kernel()", num_iter);
        for (int i = 0; i < num_iter; i++) {
            // for more sophisticated renderer (like path tracer), shared_memory should be used
            render_pt_kernel<false><<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
                *camera, *verts, obj_info, aabbs, norms, uvs, 
                bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets,
                image, output_buffer, num_prims, num_objs, num_emitter, 
                i * SEED_SCALER, max_depth, num_nodes, accum_cnt
            ); 
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
            printProgress(i, num_iter);
        }
        printf("\n");
        return image.export_cpu(1.f / num_iter, gamma_correction);
    }

    virtual CPT_CPU void render_online(
        int max_depth = 4
    ) override {
        CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &pbo_resc, 0));
        size_t _num_bytes = 0;
        CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&output_buffer, &_num_bytes, pbo_resc));

        accum_cnt ++;
        render_pt_kernel<true><<<dim3(w >> 4, h >> 4), dim3(16, 16)>>>(
            *camera, *verts, obj_info, aabbs, norms, uvs, 
            bvh_fronts, bvh_backs, node_fronts, node_backs, node_offsets,
            image, output_buffer, num_prims, num_objs, num_emitter, 
            accum_cnt * SEED_SCALER, max_depth, num_nodes, accum_cnt
        ); 
        CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &pbo_resc, 0));
    }

    /**
     * @brief initialize graphics resources
     * @param executor the callback function pointer
     */
    void graphics_resc_init(
        void (*executor) (float*, cudaGraphicsResource_t&, uint32_t&, uint32_t&, int, int)
    ) {
        executor(output_buffer, pbo_resc, pbo_id, cuda_texture_id, w, h);
    }

    CPT_CPU void update_camera(const DeviceCamera* const cam) {
        CUDA_CHECK_RETURN(cudaMemcpyAsync(camera, cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemset(image.data(), 0, w * h * sizeof(float4)));    // reset image buffer
        accum_cnt = 0;                                                                  // reset accumulation counter
    }

    CPT_CPU int get_num_sample() const noexcept {
        return accum_cnt;
    }

    CPT_CPU std::vector<uint8_t> get_image_buffer(bool gamma_cor) const {
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        return image.export_cpu(1.f / accum_cnt, gamma_cor);
    }

    template <typename TexType>
    static void createTexture1D(const TexType* tex_src, size_t size, TexType* tex_dst, cudaTextureObject_t& tex_obj) {
        cudaChannelFormatDesc channel_desc;
        if constexpr (std::is_same_v<std::decay_t<TexType>, int>) {
            channel_desc = cudaCreateChannelDesc<int>();
        } else {
            channel_desc = cudaCreateChannelDesc<float4>();
        }
        CUDA_CHECK_RETURN(cudaMemcpy(tex_dst, tex_src, size * sizeof(TexType), cudaMemcpyHostToDevice));
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeLinear;
        res_desc.res.linear.devPtr = tex_dst;
        res_desc.res.linear.desc   = channel_desc;
        res_desc.res.linear.sizeInBytes = size * sizeof(TexType);

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModePoint;
        tex_desc.readMode = cudaReadModeElementType;

        CUDA_CHECK_RETURN(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));
    }
};
