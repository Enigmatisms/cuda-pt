// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <ext/imgui/backends/imgui_impl_glfw.h>
#include <ext/imgui/backends/imgui_impl_opengl3.h>
#include <ext/imgui/imgui.h>
#include <sstream>

#include "core/imgui_utils.cuh"
#include "core/scene.cuh"
#include "core/serialize.h"
#include "core/xyz.cuh"

#include "renderer/bvh_cost.cuh"
#include "renderer/light_tracer.cuh"
#include "renderer/volume_pt.cuh"
#include "renderer/wf_path_tracer.cuh"

extern CPT_GPU_CONST Emitter *c_emitter[9];
extern CPT_GPU_CONST BSDF *c_material[48];

std::string get_current_time() {
    // Get the current time as a time_point
    auto now = std::chrono::system_clock::now();

    // Convert to time_t to extract time components
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *local_time = std::localtime(&now_time);

    // Use stringstream to format the output
    std::stringstream oss;
    oss << std::put_time(local_time, "%Y-%m-%d-%H-%M-%S");

    return oss.str();
}

int main(int argc, char **argv) {
    CUDA_CHECK_RETURN(cudaFree(nullptr)); // initialize CUDA
#ifdef OPENMP_ENABLED
    omp_set_num_threads(4);
#endif // OPENMP_ENABLED

    std::cerr << "[MAIN] Path tracing IMGUI viewer.\n";
    if (argc < 2) {
        std::cerr << "Input file not specified. Usage: ./pt <path to xml>\n";
        exit(1);
    }
    std::string xml_path = argv[1];

    std::cout << "[SCENE] Loading scenes from '" << xml_path << "'\n";
    Scene scene(xml_path);

    ColorSpaceXYZ xyz_host;
    xyz_host.init();
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, scene.bsdfs,
                                         scene.num_bsdfs * sizeof(BSDF *)));
    CUDA_CHECK_RETURN(
        cudaMemcpyToSymbol(c_emitter, scene.emitters,
                           (scene.num_emitters + 1) * sizeof(Emitter *)));
#ifdef TRIANGLE_ONLY
    printf("[ATTENTION] Note that TRIANGLE_ONLY macro is defined. Please make "
           "sure there is no sphere primitive in the scene.\n");
#endif
    std::unique_ptr<TracerBase> renderer = nullptr;
    std::cout << "[RENDERER] Path tracer loaded: ";
    switch (scene.rdr_type) {
    case RendererType::MegaKernelPT: {
        renderer = std::make_unique<PathTracer<SingleTileScheduler>>(scene);
        std::cout << "\tMegakernel Path Tracing.\n";
        break;
    }
    case RendererType::WavefrontPT: {
        renderer = std::make_unique<WavefrontPathTracer>(scene);
        std::cout << "\tWavefront Path Tracing..\n";
        break;
    }
    case RendererType::MegaKernelLT: {
        renderer = std::make_unique<LightTracer>(
            scene, scene.config.spec_constraint, scene.config.bidirectional,
            scene.config.caustic_scaling);
        if (scene.config.bidirectional)
            std::cout << "\tNaive Bidirectional Megakernel Light Tracing.\n";
        else
            std::cout << "\tMegakernel Light Tracing.\n";
        break;
    }
    case RendererType::MegaKernelVPT: {
        renderer = std::make_unique<VolumePathTracer>(scene);
        std::cout << "\tVolumetric Path Tracer\n";
        break;
    }
    case RendererType::VoxelSDFPT: {
        std::cerr
            << "\tVoxelSDFPT is not implemented yet. Stay tuned. Rendering "
               "exits.\n";
        return 0;
    }
    case RendererType::DepthTracing: {
        renderer = std::make_unique<DepthTracer>(scene);
        std::cout << "\tDepth Tracing\n";
        break;
    }
    case RendererType::BVHCostViz: {
        renderer = std::make_unique<BVHCostVisualizer>(scene);
        std::cout << "\tBVH Cost Visualizer\n";
        break;
    }
    case RendererType::MegaKernelPTDynamic: {
        renderer =
            std::make_unique<PathTracer<PreemptivePersistentTileScheduler>>(
                scene);
        std::cout << "\tMegakernel Path Tracing (Dynamic Scheduler).\n";
        break;
    }
    default: {
        std::cerr << "Renderer type: '" << RENDER_TYPE_STR[scene.rdr_type]
                  << "' unsupported for interactive viewer.\n";
        throw std::runtime_error("Unsupported renderer type.");
    }
    }

    scene.free_resources();

    auto window = gui::create_window(scene.config.width, scene.config.height);
    renderer->graphics_resc_init(gui::init_texture_and_pbo);
    renderer->update_camera(scene.cam);
    gui::GUIParams params(renderer->serialized_data);
    Serializer::push<int>(params.serialized_data, 1);
    params.gamma_corr = scene.config.gamma_correction;
    bool exit_main_loop = false;

    ImGuiIO &io = ImGui::GetIO();

    while (!glfwWindowShouldClose(window.get())) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        gui::show_render_statistics(renderer->get_num_sample() + 1,
                                    params.show_fps);
        params.reset();
        params.camera_update = gui::keyboard_camera_update(
            *scene.cam, params.trans_speed, params.capture, exit_main_loop);
        if (exit_main_loop) {
            break;
        }
        gui::render_settings_interface(*scene.cam, scene.emitter_props,
                                       scene.bsdf_infos, scene.medium_infos,
                                       scene.config.md, params, scene.rdr_type);
        if (!io.WantCaptureMouse) { // no sub window (setting window or main
                                    // menu) is focused
            params.camera_update |=
                gui::mouse_camera_update(*scene.cam, params.rot_sensitivity);
        }
        if (params.scene_update) {
            scene.update_emitters();
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, scene.emitters,
                                                 (scene.num_emitters + 1) *
                                                     sizeof(Emitter *)));
        }
        if (params.material_update) {
            scene.update_materials();
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(
                c_material, scene.bsdfs, scene.num_bsdfs * sizeof(BSDF *)));
        }
        if (params.medium_update) {
            scene.update_media();
        }
        if (params.camera_update) {
            renderer->update_camera(scene.cam);
        }
        if (params.serialized_update) {
            renderer->param_setter(params.serialized_data);
        }
        if (params.buffer_flush_update())
            renderer->reset_out_buffer();
        renderer->render_online(scene.config.md, params.gamma_corr);

        if (params.capture) {
            auto fbuffer = renderer->get_image_buffer(params.gamma_corr);
            std::string format = params.output_png ? "png" : "jpg";
            std::string file_name =
                "render-" + get_current_time() + "." + format;

            if (!save_image(file_name, fbuffer, scene.config.width,
                            scene.config.height, format, params.compress_q)) {
                std::cerr << "stb::save_image() failed to output image"
                          << std::endl;
                throw std::runtime_error("stb::save_image() fail");
            } else {
                std::cout << "[Viewer] Image file saved to '" << file_name
                          << "'\n";
            }
        }
        gui::update_texture(renderer->get_pbo_id(), renderer->get_texture_id(),
                            scene.config.width, scene.config.height);
        gui::window_render(renderer->get_texture_id(), scene.config.width,
                           scene.config.height);

        // swap the buffer
        glfwSwapBuffers(window.get());
    }
    scene.print();
    gui::clean_up(window.get(), renderer->get_pbo_id(),
                  renderer->get_texture_id());
    xyz_host.destroy();

    return 0;
}
