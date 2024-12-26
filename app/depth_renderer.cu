#include "core/aos.cuh"
#include "renderer/depth.cuh"
#include "core/camera_model.cuh"
#include "core/scene.cuh"

int main(int argc, char** argv) {
    CUDA_CHECK_RETURN(cudaFree(nullptr));       // initialize CUDA
    if (argc < 2) {
        std::cerr << "Input file not specified. Usage: ./pt <path to xml>\n";
        exit(1);
    }
    std::string xml_path = argv[1];

    std::cout << "Loading scenes from '" << xml_path << "'\n";
    Scene scene(xml_path);

    DepthTracer dtracer(scene);
    auto bytes_buffer = dtracer.render(16);

    std::string file_name = "depth-render.png";
    if (!save_image(file_name, bytes_buffer, scene.config.width, scene.config.height, "png")) {
        std::cerr << "stb::save_image() failed to output image" << std::endl;
        throw std::runtime_error("stb::save_image() fail");
    }

    printf("image saved to `%s`\n", file_name.c_str());

    return 0;
}