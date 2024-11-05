#include "core/aos.cuh"
#include "renderer/depth.cuh"
#include "core/camera_model.cuh"
#include "core/scene.cuh"
#include <ext/lodepng/lodepng.h>

int main(int argc, char** argv) {
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

    if (unsigned error = lodepng::encode(file_name, bytes_buffer, scene.config.width, scene.config.height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }

    printf("image saved to `%s`\n", file_name.c_str());

    return 0;
}