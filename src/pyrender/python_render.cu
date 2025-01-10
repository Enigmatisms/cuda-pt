/**
 * @file python_render.cpp
 * @author Qianyue He
 * @brief Renderer Nanobind bindings
 * @date 2025-01-10
 * @copyright Copyright (c) 2025
 */

#include "./python_render.cuh"
#include "core/serialize.h"
#include "renderer/bvh_cost.cuh"
#include "renderer/light_tracer.cuh"
#include "renderer/wf_path_tracer.cuh"
#include "core/scene.cuh"

nb::ndarray<nb::numpy, float> PythonRenderer::render(
    int max_bounce,
    int max_diffuse,
    int max_specular,
    int max_trans,
    bool gamma_corr
) {
    MaxDepthParams md_params(max_bounce, max_diffuse, max_specular, max_trans);
    float* gpu_ptr = rdr->render_raw(md_params, gamma_corr);
    return {};
}

PythonRenderer::PythonRenderer(const nb::str& xml_path) {
    xyz_host = std::make_unique<ColorSpaceXYZ>();
    std::string path = std::string(xml_path.c_str());
    scene = std::make_unique<Scene>(path);
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
    rdr->update_camera(scene->cam);
    rdr->initialize_output_buffer();
}

void PythonRenderer::release() {
    scene->print();
    xyz_host->destroy();
}
