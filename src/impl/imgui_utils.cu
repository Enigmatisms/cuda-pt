#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <ext/imgui/imgui.h>
#include <ext/imgui/backends/imgui_impl_glfw.h>
#include <ext/imgui/backends/imgui_impl_opengl3.h>
#include <cuda_gl_interop.h>
#include "core/cuda_utils.cuh"
#include "core/imgui_utils.cuh"

namespace gui {

// initialize GL texture and PBO (pixel buffer object)
void init_gl_texture_and_pbo(
    float* output_buffer,
    cudaGraphicsResource_t& pbo_resc, 
    GLuint& pbo_id, GLuint& cuda_texture_id,
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

void imgui_window_render(GLuint cuda_texture_id, int width, int height) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("CUDA-PT Rendering Output");
    ImGui::Image((void*)(intptr_t)cuda_texture_id, ImVec2(width, height));
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void gl_update_texture(
    GLuint pbo_id, GLuint cuda_texture_id,
    int width, int height
) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
    glBindTexture(GL_TEXTURE_2D, cuda_texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void imgui_clean_up(
    GLFWwindow* window,
    GLuint& pbo_id, GLuint& cuda_texture_id
) {
    glDeleteBuffers(1, &pbo_id);
    glDeleteTextures(1, &cuda_texture_id);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

std::unique_ptr<GLFWwindow, GLFWWindowDeleter> create_imgui_window(int width, int height) {
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

}   // namespace gui