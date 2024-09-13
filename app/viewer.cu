#include "core/scene.cuh"
#include "renderer/wf_path_tracer.cuh"
#include <ext/lodepng/lodepng.h>
#include <ext/imgui/imgui.h>
#include <ext/imgui/backends/imgui_impl_glfw.h>
#include <ext/imgui/backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>


GLuint my_render_texture; // 这是你的渲染结果纹理

void RenderImGuiWindow() {
    // 绑定 ImGui 帧
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 创建一个新的 ImGui 窗口
    ImGui::Begin("Rendering Output");

    // 使用 ImGui::Image 显示 OpenGL 纹理
    // 纹理 ID 需要转换为 (void*) 类型
    ImGui::Image((void*)(intptr_t)my_render_texture, ImVec2(800, 600)); // 调整尺寸到你的纹理大小

    ImGui::End();

    // 渲染 ImGui 内容
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


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

    scene.export_prims(vert_data, norm_data, uvs_data);

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, scene.bsdfs, scene.num_bsdfs * sizeof(BSDF*)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, scene.emitters, (scene.num_emitters + 1) * sizeof(Emitter*)));

    std::unique_ptr<PathTracer> renderer = nullptr;
    if (scene.rdr_type == RendererType::MegaKernelPT) {
        renderer = std::make_unique<PathTracer>(scene, vert_data, norm_data, uvs_data, 1);
    } else {
        renderer = std::make_unique<WavefrontPathTracer>(scene, vert_data, norm_data, uvs_data, 1);
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // 主循环
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // 渲染你的 GPU 光追输出到 my_render_texture
        // RenderYourRayTracingOutputToTexture(my_render_texture);

        // 使用 ImGui 显示渲染结果
        RenderImGuiWindow();

        // 交换缓冲区，显示图像
        glfwSwapBuffers(window);
    }

    // 清理 ImGui 和 OpenGL 资源
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

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