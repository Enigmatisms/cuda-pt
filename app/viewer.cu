#include <sstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <ext/imgui/imgui.h>
#include <ext/lodepng/lodepng.h>
#include <ext/imgui/backends/imgui_impl_glfw.h>
#include <ext/imgui/backends/imgui_impl_opengl3.h>

#include "core/scene.cuh"
#include "core/imgui_utils.cuh"
#include "renderer/light_tracer.cuh"
#include "renderer/wf_path_tracer.cuh"

__constant__ Emitter* c_emitter[9];
__constant__ BSDF*    c_material[32];

std::string get_current_time() {
    // Get the current time as a time_point
    auto now = std::chrono::system_clock::now();

    // Convert to time_t to extract time components
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);

    // Use stringstream to format the output
    std::stringstream oss;
    oss << std::put_time(local_time, "%Y-%m-%d-%H-%M-%S");

    return oss.str();
}

int main(int argc, char** argv) {
    CUDA_CHECK_RETURN(cudaFree(nullptr));       // initialize CUDA
    std::cerr << "[MAIN] Path tracing IMGUI viewer.\n";
    if (argc < 2) {
        std::cerr << "Input file not specified. Usage: ./pt <path to xml>\n";
        exit(1);
    }
    std::string xml_path = argv[1];

    std::cout << "[SCENE] Loading scenes from '" << xml_path << "'\n";
    Scene scene(xml_path);

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, scene.bsdfs, scene.num_bsdfs * sizeof(BSDF*)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, scene.emitters, (scene.num_emitters + 1) * sizeof(Emitter*)));

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
            std::cout << "\tWavefront Path Tracing..\n";
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

    auto window = gui::create_window(scene.config.width, scene.config.height);
    renderer->graphics_resc_init(gui::init_texture_and_pbo);
    renderer->update_camera(scene.cam);

    bool show_main_settings  = true;
    bool show_frame_rate_bar = true;
    bool frame_capture       = false;
    bool exit_main_loop      = false;
    bool gamma_correct       = true;
    ImGuiIO& io = ImGui::GetIO();

    while (!glfwWindowShouldClose(window.get())) {
        frame_capture = false;
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        gui::show_render_statistics(
            renderer->get_num_sample() + 1,
            show_frame_rate_bar
        );
        bool cam_updated = gui::keyboard_camera_update(*scene.cam, 0.1, frame_capture, exit_main_loop),
            scene_updated = false, 
            renderer_update = false,
            material_update = false;
        if (exit_main_loop) {
            break;
        }
        gui::render_settings_interface(
            *scene.cam, scene.emitter_props, scene.bsdf_infos, scene.config.max_depth, 
            show_main_settings, show_frame_rate_bar, frame_capture, 
            gamma_correct, cam_updated, scene_updated, material_update, renderer_update
        );
        if (!io.WantCaptureMouse) {        // no sub window (setting window or main menu) is focused
            cam_updated |= gui::mouse_camera_update(*scene.cam, 0.5);
        }
        if (scene_updated) {
            scene.update_emitters();
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, scene.emitters, (scene.num_emitters + 1) * sizeof(Emitter*)));
        }
        if (material_update) {
            scene.update_materials();
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, scene.bsdfs, scene.num_bsdfs * sizeof(BSDF*)));
        }
        if (cam_updated) {
            renderer->update_camera(scene.cam);
        }
        if (cam_updated || scene_updated || material_update || renderer_update)
            renderer->reset_out_buffer();
        renderer->render_online(scene.config.max_depth, gamma_correct);
        
        if (frame_capture) {
            auto fbuffer = renderer->get_image_buffer(gamma_correct);
            std::string file_name = "render-" + get_current_time() + ".png";
            if (unsigned error = lodepng::encode(file_name, fbuffer, scene.config.width, scene.config.height); error) {
                std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                        << std::endl;
                throw std::runtime_error("lodepng::encode() fail");
            } else {
                std::cout << "[Viewer] Image file saved to '" << file_name << "'\n";
            }
        }
        gui::update_texture(
            renderer->get_pbo_id(),
            renderer->get_texture_id(),
            scene.config.width,
            scene.config.height
        );
        gui::window_render(
            renderer->get_texture_id(),
            scene.config.width,
            scene.config.height
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

    return 0;
}