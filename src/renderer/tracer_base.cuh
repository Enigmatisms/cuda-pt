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

    DeviceCamera* camera;
    float* output_buffer;                // output buffer for images
    
    // IMGUI related 
    int accum_cnt;
    uint32_t cuda_texture_id, pbo_id;
    cudaGraphicsResource_t pbo_resc;
public:
    /**
     * @param shapes    shape information (for AABB generation)
     * @param verts     vertices, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param norms     normal vectors, ArrayType: (p1, 3D) -> (p2, 3D) -> (p3, 3D)
     * @param uvs       uv coordinates, ArrayType: (p1, 2D) -> (p2, 2D) -> (p3, 2D)
     * @param camera    GPU camera model (constant memory)
     * @param image     GPU image buffer
     * @param skip_vertex_loading For OptiX renderer, if true, verts will be destroyed
    */
    TracerBase(
        const Scene& scene,
        bool skip_vertex_loading = false
    ): verts(scene.num_prims), norms(scene.num_prims), uvs(scene.num_prims),
       image(scene.config.width, scene.config.height), 
       num_prims(scene.num_prims), 
       w(scene.config.width), 
       h(scene.config.height),
       output_buffer(nullptr),
       accum_cnt(0),
       cuda_texture_id(0), pbo_id(0)
    {
        // TODO: shapes is so fucking useless
        scene.export_prims(verts, norms, uvs);
        if (skip_vertex_loading) {
            aabbs = nullptr;
        } else {
            CUDA_CHECK_RETURN(cudaMallocManaged(&aabbs, num_prims * sizeof(AABB)));
            ShapeAABBVisitor aabb_visitor(verts, aabbs);
            // calculate AABB for each primitive
            for (int i = 0; i < num_prims; i++) {
                aabb_visitor.set_index(i);
                std::visit(aabb_visitor, scene.shapes[i]);
            }
        }
    }

    virtual ~TracerBase() {
        CUDA_CHECK_RETURN(cudaFree(aabbs));     // free nullptr is legal
        image.destroy();
        verts.destroy();
        norms.destroy();
        uvs.destroy();
    }

    CPT_CPU uint32_t& get_texture_id() noexcept { return this->cuda_texture_id; }
    CPT_CPU uint32_t& get_pbo_id()     noexcept { return this->pbo_id; }

    CPT_CPU virtual std::vector<uint8_t> render(
        int num_iter  = 64,
        int max_depth = 1,/* max depth, useless for depth renderer, 1 anyway */
        bool gamma_correction = true
    ) {
        throw std::runtime_error("Not implemented.\n");
        return {};
    }

    CPT_CPU virtual void render_online(
        int max_depth = 1, /* max depth, useless for depth renderer, 1 anyway */
        bool gamma_corr = false     /* whether to enable gamma correction*/
    ) {
        throw std::runtime_error("Not implemented.\n");
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

    CPT_CPU_INLINE void reset_out_buffer() {
        CUDA_CHECK_RETURN(cudaMemset(image.data(), 0, w * h * sizeof(float4)));    // reset image buffer
        accum_cnt = 0;                                                              // reset accumulation counter
    }

    CPT_CPU void update_camera(const DeviceCamera* const cam) {
        CUDA_CHECK_RETURN(cudaMemcpyAsync(camera, cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice));
    }

    CPT_CPU int get_num_sample() const noexcept {
        return accum_cnt;
    }

    CPT_CPU std::vector<uint8_t> get_image_buffer(bool gamma_cor) const {
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        return image.export_cpu(1.f / accum_cnt, gamma_cor);
    }
};
