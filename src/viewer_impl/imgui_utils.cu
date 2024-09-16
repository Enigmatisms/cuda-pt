#include <GL/glew.h>
#include <ext/imgui/imgui.h>
#include <ext/imgui/backends/imgui_impl_glfw.h>
#include <ext/imgui/backends/imgui_impl_opengl3.h>
#include <cuda_gl_interop.h>
#include "core/cuda_utils.cuh"
#include "core/camera_model.cuh"
#include "core/imgui_utils.cuh"

namespace gui {

// initialize GL texture and PBO (pixel buffer object)
void init_texture_and_pbo(
    float* output_buffer,
    cudaGraphicsResource_t& pbo_resc, 
    gl_uint& pbo_id, gl_uint& cuda_texture_id,
    int width, int height
) {
    // create PBO
    glGenBuffers(1, &pbo_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(float), 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register PBO
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(&pbo_resc, pbo_id, cudaGraphicsMapFlagsWriteDiscard));

    // Create OpenGL texture (context)
    glGenTextures(1, &cuda_texture_id);
    glBindTexture(GL_TEXTURE_2D, cuda_texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

    // Allocate accumulation buffer
    CUDA_CHECK_RETURN(cudaMalloc(&output_buffer, width * height * 4 * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemset(output_buffer, 0, width * height * 4 * sizeof(float)));
}

void sub_window_render(std::string sub_win_name, gl_uint cuda_texture_id, int width, int height) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin(sub_win_name.c_str());
    ImGui::Image((void*)(intptr_t)cuda_texture_id, ImVec2(width, height));
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void window_render(
    GLuint cuda_texture_id, int width, int height,
    bool show_fps
) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImVec2 top_left(0, 0);
    ImVec2 bottom_right(width, height);

    // splat pixel buffer outputs
    draw_list->AddImage((void*)(intptr_t)cuda_texture_id, top_left, bottom_right);
    if (show_fps) {
        const char* window_title = "Statistics";
        ImVec2 text_size = ImGui::CalcTextSize(window_title);
        float padding = ImGui::GetStyle().WindowPadding.x * 2;
        float min_width = text_size.x * 1.2f + padding; 
        ImGui::SetNextWindowSizeConstraints(ImVec2(min_width, 0), ImVec2(FLT_MAX, FLT_MAX));
        ImGui::Begin(window_title, nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        ImGuiIO& io = ImGui::GetIO();
        ImGui::Text("FPS: %.2f", io.Framerate);
        ImGui::End();
    }
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void update_texture(
    gl_uint pbo_id, gl_uint cuda_texture_id,
    int width, int height
) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
    glBindTexture(GL_TEXTURE_2D, cuda_texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void clean_up(
    GLFWwindow* window,
    gl_uint& pbo_id, gl_uint& cuda_texture_id
) {
    glDeleteBuffers(1, &pbo_id);
    glDeleteTextures(1, &cuda_texture_id);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

std::unique_ptr<GLFWwindow, GLFWWindowDeleter> create_window(int width, int height) {
    // initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // OPENGL window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA-PT", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vertical sync

    // initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return nullptr;
    }

    // initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    return std::unique_ptr<GLFWwindow, GLFWWindowDeleter>(window);
}

bool keyboard_camera_update(DeviceCamera& camera, float step)
{
    ImGuiIO& io = ImGui::GetIO();
    bool update = false;
    if (io.KeysDown[ImGuiKey_W]) {
        update = true;
        camera.move_forward(step);
    }
    if (io.KeysDown[ImGuiKey_S]) {
        update = true;
        camera.move_backward(step);
    }
    if (io.KeysDown[ImGuiKey_A]) {
        update = true;
        camera.move_left(step);
    }
    if (io.KeysDown[ImGuiKey_D]) {
        update = true;
        camera.move_right(step);
    }
    return update;
}

}   // namespace gui