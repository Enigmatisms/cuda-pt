#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <ext/imgui/imgui.h>
#include <ext/imgui/backends/imgui_impl_glfw.h>
#include <ext/imgui/backends/imgui_impl_opengl3.h>

#include "core/scene.cuh"
#include "core/imgui_utils.cuh"
#include "renderer/wf_path_tracer.cuh"

__constant__ Emitter* c_emitter[9];
__constant__ BSDF*    c_material[32];

int main(int argc, char** argv) {
    std::cerr << "Path tracing IMGUI viewer.\n";
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

    scene.export_prims(vert_data, norm_data, uvs_data);

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, scene.bsdfs, scene.num_bsdfs * sizeof(BSDF*)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, scene.emitters, (scene.num_emitters + 1) * sizeof(Emitter*)));

    std::unique_ptr<PathTracer> renderer = nullptr;
    std::cout << "Path tracer loaded: ";
    if (scene.rdr_type == RendererType::MegaKernelPT) {
        renderer = std::make_unique<PathTracer>(scene, vert_data, norm_data, uvs_data, 1);
        std::cout << "Megakernel PT.\n";
    } else {
        renderer = std::make_unique<WavefrontPathTracer>(scene, vert_data, norm_data, uvs_data, 1);
        std::cout << "Wavefront PT.\n";
    }

    auto window = gui::create_window(scene.config.width, scene.config.height);
    renderer->graphics_resc_init(gui::init_texture_and_pbo);
    renderer->update_camera(scene.cam);

    while (!glfwWindowShouldClose(window.get())) {
        glfwPollEvents();
        bool cam_updated = gui::keyboard_camera_update(*scene.cam, 0.1);
        if (cam_updated) {
            renderer->update_camera(scene.cam);
        }
        renderer->render_online(scene.config.max_depth);
        gui::update_texture(
            renderer->get_pbo_id(),
            renderer->get_texture_id(),
            scene.config.width,
            scene.config.height
        );
        gui::window_render(
            renderer->get_texture_id(),
            scene.config.width,
            scene.config.height,
            true
        );

        // swap the buffer
        glfwSwapBuffers(window.get());
    }
    scene.print();

    gui::clean_up(
        window.get(),
        renderer->get_pbo_id(),
        renderer->get_texture_id()
    );

    vert_data.destroy();
    norm_data.destroy();
    uvs_data.destroy();

    return 0;
}