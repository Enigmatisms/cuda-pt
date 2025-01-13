/**
 * @file python_render.cpp
 * @author Qianyue He
 * @brief Renderer Nanobind bindings
 * @date 2025-01-10
 * @copyright Copyright (c) 2025
 */
#include "./python_render.cuh"
#include "core/stats.h"
#include "core/scene.cuh"
#include "core/serialize.h"
#include "renderer/bvh_cost.cuh"
#include "renderer/light_tracer.cuh"
#include "renderer/wf_path_tracer.cuh"

static nb::ndarray<nb::pytorch, float> gpu_ndarray_deep_copy(float* gpu_src_ptr, size_t width, size_t height, int dev_id = 0) {
    int num_elements = width * height * 4;

    float* gpu_dst_ptr;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&gpu_dst_ptr, num_elements * sizeof(float)));

    nb::capsule deleter(gpu_dst_ptr, [](void *p) noexcept {
        CUDA_CHECK_RETURN(cudaFree(p));
    });

    CUDA_CHECK_RETURN(cudaMemcpy(gpu_dst_ptr, gpu_src_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    return nb::ndarray<nb::pytorch, float>(gpu_dst_ptr, {height, width, 4}, deleter, {}, nb::dtype<float>(), nb::device::cuda::value, dev_id);
}

nb::ndarray<nb::pytorch, float> PythonRenderer::render(
    int max_bounce,
    int max_diffuse,
    int max_specular,
    int max_trans,
    bool gamma_corr
) {
    MaxDepthParams md_params(max_diffuse, max_specular, max_trans, max_bounce);
    TicTocLocal timer;
    float* gpu_ptr = rdr->render_raw(md_params, gamma_corr);
    ftimer->record(timer.toc());
    return gpu_ndarray_deep_copy(gpu_ptr, rdr->width(), rdr->height(), device_id);
}

PythonRenderer::PythonRenderer(const nb::str& xml_path, int _device_id, int seed_offset): valid(true), device_id(_device_id) {
    CUDA_CHECK_RETURN(cudaSetDevice(_device_id));
    CUDA_CHECK_RETURN(cudaFree(nullptr));           // initialize CUDA

    std::string path = std::string(xml_path.c_str());
    ftimer   = std::make_unique<SlidingWindowAverage>(32);
    xyz_host = std::make_unique<ColorSpaceXYZ>();
    scene    = std::make_unique<Scene>(path);

    xyz_host->init();
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, scene->bsdfs, scene->num_bsdfs * sizeof(BSDF*)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, scene->emitters, (scene->num_emitters + 1) * sizeof(Emitter*)));

    std::cout << "[RENDERER] Path tracer loaded: ";
    switch (scene->rdr_type) {
        case RendererType::MegaKernelPT: {
            rdr = std::make_unique<PathTracer>(*scene); 
            std::cout << "\tMegakernel Path Tracing.\n";
            break;
        }
        case RendererType::WavefrontPT: {
            rdr = std::make_unique<WavefrontPathTracer>(*scene);
            std::cout << "\tWavefront Path Tracing..\n";
            break;
        }
        case RendererType::MegeKernelLT: {
            rdr = std::make_unique<LightTracer>(*scene, scene->config.spec_constraint,
                        scene->config.bidirectional, scene->config.caustic_scaling); 
            if (scene->config.bidirectional)
                std::cout << "\tNaive Bidirectional Megakernel Light Tracing.\n";
            else
                std::cout << "\tMegakernel Light Tracing.\n";
            break;
        } 
        case RendererType::VoxelSDFPT: {
            std::cerr << "\tVoxelSDFPT is not implemented yet. Stay tuned. Rendering exits.\n";
            exit(0);
        }
        case RendererType::DepthTracing: {
            rdr = std::make_unique<DepthTracer>(*scene);
            std::cerr << "\tDepth Tracing\n";
            break;
        }
        case RendererType::BVHCostViz: {
            rdr = std::make_unique<BVHCostVisualizer>(*scene);
            std::cerr << "\tBVH Cost Visualizer\n";
            break;
        }
        default: {
            throw std::runtime_error("Unsupported renderer type.");
        }
    }
    scene->free_resources();
    rdr->set_seed_offset(seed_offset);
    rdr->update_camera(scene->cam);
    rdr->initialize_output_buffer();
}

void PythonRenderer::release() {
    xyz_host->destroy();
    valid = false;
}

void PythonRenderer::info() const {
    scene->print();
}