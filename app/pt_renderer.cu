#include "core/scene.cuh"
#include "renderer/light_tracer.cuh"
#include "renderer/wf_path_tracer.cuh"
#include <ext/lodepng/lodepng.h>

__constant__ Emitter* c_emitter[9];
__constant__ BSDF*    c_material[32];

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Input file not specified. Usage: ./pt <path to xml>\n";
        exit(1);
    }
    std::string xml_path = argv[1];

    std::cout << "Loading scenes from '" << xml_path << "'\n";
    Scene scene(xml_path);

    // scene setup
    PrecomputedArray vert_data(scene.num_prims);
    ArrayType<Vec3> norm_data(scene.num_prims);
    ArrayType<Vec2> uvs_data(scene.num_prims);

    scene.export_prims(vert_data, norm_data, uvs_data);

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, scene.bsdfs, scene.num_bsdfs * sizeof(BSDF*)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, scene.emitters, (scene.num_emitters + 1) * sizeof(Emitter*)));

    std::unique_ptr<PathTracer> renderer = nullptr;
    std::cout << "Path tracer loaded: ";
    switch (scene.rdr_type) {
        case RendererType::MegaKernelPT: {
            renderer = std::make_unique<PathTracer>(scene, vert_data, norm_data, uvs_data, scene.num_emitters); 
            std::cout << "Megakernel Path Tracing.\n";
            break;
        }
        case RendererType::WavefrontPT: {
            renderer = std::make_unique<WavefrontPathTracer>(scene, vert_data, norm_data, uvs_data, scene.num_emitters);
            std::cout << "Wavefront Path Tracing.\n";
            break;
        }
        case RendererType::MegeKernelLT: {
            renderer = std::make_unique<LightTracer>(scene, vert_data, norm_data, uvs_data, scene.num_emitters, 
                scene.config.spec_constraint, scene.config.bidirectional, scene.config.caustic_scaling); 
            if (scene.config.bidirectional)
                std::cout << "Naive Bidirectional ";
            std::cout << "Megakernel Light Tracing.\n";
            break;
        } 
        case RendererType::VoxelSDFPT: {
            std::cerr << "VoxelSDFPT is not implemented yet. Stay tuned. Rendering exits.\n";
            return 0;
        }
        default: {
            throw std::runtime_error("Unsupported renderer type.");
        }
    }
    renderer->update_camera(scene.cam);

    printf("Prepare to render the scene... [%d] bounces, [%d] SPP\n", scene.config.max_depth, scene.config.spp);
    auto bytes_buffer = renderer->render(scene.config.spp, scene.config.max_depth, scene.config.gamma_correction);

    std::string file_name = "render.png";

    if (unsigned error = lodepng::encode(file_name, bytes_buffer, scene.config.width, scene.config.height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }

    printf("image saved to `%s`\n", file_name.c_str());
    scene.print();

    vert_data.destroy();
    norm_data.destroy();
    uvs_data.destroy();

    return 0;
}