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

static constexpr int SEED_SCALER = 11451;       //-4!

class PathTracer: public TracerBase {
private:
    int* _obj_idxs;
    float4* _node_fronts;
    float4* _node_backs;
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
    int num_cache;                  // number of cached BVH nodes
    int num_emitter;

    cudaTextureObject_t bvh_leaves;
    cudaTextureObject_t node_fronts;
    cudaTextureObject_t node_backs;
    float4* _cached_nodes;

    DeviceCamera* camera;

    // IMGUI related 
    uint32_t cuda_texture_id, pbo_id;
    cudaGraphicsResource_t pbo_resc;
    float* output_buffer;                // output buffer for images
    int* emitter_prims;
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
        const PrecomputedArray& _verts,
        const ArrayType<Vec3>& _norms, 
        const ConstBuffer<PackedHalf2>& _uvs,
        int num_emitter
    );

    virtual ~PathTracer();

    CPT_CPU uint32_t& get_texture_id() noexcept { return this->cuda_texture_id; }
    CPT_CPU uint32_t& get_pbo_id()     noexcept { return this->pbo_id; }

    virtual CPT_CPU std::vector<uint8_t> render(
        int num_iter = 64,
        int max_depth = 4,
        bool gamma_correction = true
    ) override;

    virtual CPT_CPU void render_online(
        int max_depth = 4
    ) override;

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
