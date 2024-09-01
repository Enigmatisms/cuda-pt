#include "core/scene.cuh"
#include "renderer/wf_path_tracer.cuh"
#include <ext/lodepng/lodepng.h>

__constant__ DeviceCamera dev_cam;
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
    ArrayType<Vec3> vert_data(scene.num_prims), norm_data(scene.num_prims);
    ArrayType<Vec2> uvs_data(scene.num_prims);

    scene.export(vert_data, norm_data, uvs_data);

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, scene.bsdfs, scene.num_bsdfs * sizeof(BSDF*)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, scene.emitters, (scene.num_emitters + 1) * sizeof(Emitter*)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(dev_cam, &scene.cam, sizeof(DeviceCamera)));

    std::unique_ptr<PathTracer> renderer = nullptr;
    if (scene.rdr_type == RendererType::MegaKernelPT) {
        renderer = std::make_unique<PathTracer>(scene, vert_data, norm_data, uvs_data, 1);
    } else {
        renderer = std::make_unique<WavefrontPathTracer>(scene, vert_data, norm_data, uvs_data, 1);
    }

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