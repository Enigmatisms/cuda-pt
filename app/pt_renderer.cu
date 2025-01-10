#include "core/scene.cuh"
#include "renderer/light_tracer.cuh"
#include "renderer/wf_path_tracer.cuh"

extern CPT_GPU_CONST Emitter* c_emitter[9];
extern CPT_GPU_CONST BSDF*    c_material[48];

int main(int argc, char** argv) {
    CUDA_CHECK_RETURN(cudaFree(nullptr));       // initialize CUDA
    if (argc < 2) {
        std::cerr << "Input file not specified. Usage: ./pt <path to xml>\n";
        exit(1);
    }
    std::string xml_path = argv[1];

    std::cout << "[SCENE] Loading scenes from '" << xml_path << "'\n";
    Scene scene(xml_path);

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, scene.bsdfs, scene.num_bsdfs * sizeof(BSDF*)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, scene.emitters, (scene.num_emitters + 1) * sizeof(Emitter*)));
#ifdef TRIANGLE_ONLY
    printf("[ATTENTION] Note that TRIANGLE_ONLY macro is defined. Please make sure there is no sphere primitive in the scene.\n");
#endif
    std::unique_ptr<TracerBase> renderer = nullptr;
    std::cout << "[RENDERER] Path tracer loaded: ";
    switch (scene.rdr_type) {
        case RendererType::MegaKernelPT: {
            renderer = std::make_unique<PathTracer>(scene); 
            std::cout << "\tMegakernel Path Tracing.\n";
            break;
        }
        case RendererType::WavefrontPT: {
            renderer = std::make_unique<WavefrontPathTracer>(scene);
            std::cout << "\tWavefront Path Tracing.\n";
            break;
        }
        case RendererType::MegeKernelLT: {
            renderer = std::make_unique<LightTracer>(scene, scene.config.spec_constraint, 
                    scene.config.bidirectional, scene.config.caustic_scaling); 
            if (scene.config.bidirectional)
                std::cout << "\tNaive Bidirectional Megakernel Light Tracing.\n";
            else
                std::cout << "\tMegakernel Light Tracing.\n";
            break;
        } 
        case RendererType::VoxelSDFPT: {
            std::cerr << "\tVoxelSDFPT is not implemented yet. Stay tuned. Rendering exits.\n";
            return 0;
        }
        default: {
            throw std::runtime_error("Unsupported renderer type.");
        }
    }
    renderer->update_camera(scene.cam);

    printf("[RENDERER] Prepare to render the scene... [%d] bounces, [%d] SPP\n", scene.config.md.max_depth, scene.config.spp);
    auto bytes_buffer = renderer->render(scene.config.md, scene.config.spp, scene.config.gamma_correction);

    std::string file_name = "render.png";
    if (!save_image(file_name, bytes_buffer, scene.config.width, scene.config.height, "png")) {
        std::cerr << "stb::save_image() failed to output image" << std::endl;
        throw std::runtime_error("stb::save_image() fail");
    }

    printf("[IMAGE] Image saved to `%s`\n", file_name.c_str());
    scene.print();
    return 0;
}