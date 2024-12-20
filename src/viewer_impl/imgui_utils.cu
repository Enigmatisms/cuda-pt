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
    GLuint cuda_texture_id, 
    int width, 
    int height
) {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImVec2 top_left(0, 0);
    ImVec2 bottom_right(width, height);
    // splat pixel buffer outputs
    draw_list->AddImage((void*)(intptr_t)cuda_texture_id, top_left, bottom_right);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void show_render_statistics(
    int num_sample, 
    bool& sub_window_proc,
    bool show_fps
) {
    sub_window_proc = false;
    if (show_fps) {
        const char* window_title = "Statistics";
        ImVec2 text_size = ImGui::CalcTextSize(window_title);
        float padding = ImGui::GetStyle().WindowPadding.x * 2;
        float min_width = text_size.x * 1.2f + padding; 
        ImGui::SetNextWindowSizeConstraints(ImVec2(min_width, 0), ImVec2(FLT_MAX, FLT_MAX));
        ImGui::Begin(window_title, &show_fps);
        ImGuiIO& io = ImGui::GetIO();
        ImGui::Text("FPS: %.2f", io.Framerate);
        ImGui::Text("SPP: %4d", num_sample);
        if (ImGui::IsWindowFocused()) {
            sub_window_proc = true;
        }
        ImGui::End();
    }
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

bool keyboard_camera_update(DeviceCamera& camera, float step, bool& frame_cap, bool& exiting)
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
    if (io.KeysDown[ImGuiKey_P]) {
        frame_cap = true;
        printf("Frame capture keyboard event.\n");
    }
    if (io.KeysDown[ImGuiKey_Escape]) {
        exiting = true;
    }
    if (io.KeysDown[ImGuiKey_E]) {
        printf("Camera Pose:\n");
        Vec3 lookat = camera.R.col(2) + camera.t;
        printf("\tPosition:\t");
        print_vec3(camera.t);
        printf("\tLookat:  \t");
        print_vec3(lookat);
        printf("\tUp:      \t");
        print_vec3(camera.R.col(1));
        printf("\tLateral: \t");
        print_vec3(camera.R.col(0));
        printf("\n");
    }
    return update;
}

bool mouse_camera_update(DeviceCamera& cam, float sensitivity) {
    ImGuiIO& io = ImGui::GetIO();
    // Left button is pressed?
    if (io.MouseDown[0]) {
        // get current mouse pos

        // whether the mouse is being dragged
        if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) {
            float deltaX = io.MouseDelta.x;
            float deltaY = io.MouseDelta.y;

            float yaw = deltaX * sensitivity * M_Pi / 180.0f;
            float pitch = deltaY * sensitivity * M_Pi / 180.0f;

            cam.rotate(yaw, pitch);
            return true;
        }
    }
    return false;
}

bool draw_rgb_ui(std::string description, float& r, float& g, float& b, float& scaler, bool add_rule = true)
{
    bool updated = false;
    if (add_rule)
        ImGui::Separator();

    ImGui::Text("Emission for '%s'", description.c_str());
    
    ImGui::Text("RGB");
    ImGui::SameLine();

    ImGui::PushItemWidth(100.0f); // width of the input item
    std::string label = "R##RGB-" + description;
    updated |= ImGui::InputFloat(label.c_str(), &r, 0.01f, 1.0f, "%.2f"); ImGui::SameLine();
    label = "G##RGB-" + description;
    updated |= ImGui::InputFloat(label.c_str(), &g, 0.01f, 1.0f, "%.2f"); ImGui::SameLine();
    label = "B##RGB-" + description;
    updated |= ImGui::InputFloat(label.c_str(), &b, 0.01f, 1.0f, "%.2f");
    ImGui::PopItemWidth();
    ImGui::Text("Scaler");
    ImGui::SameLine();

    ImGui::PushItemWidth(120.0f);
    label = "##ScalerSlider-" + description;
    updated |= ImGui::SliderFloat(label.c_str(), &scaler, 0.0f, 100.0f, "%.2f"); ImGui::SameLine();
    label = "##ScalerInput-" + description;
    updated |= ImGui::InputFloat(label.c_str(), &scaler, 0.0f, 100.0f, "%.2f");
    ImGui::PopItemWidth();
    return updated;
}

void render_settings_interface(
    DeviceCamera& cam, 
    std::vector<std::pair<std::string, Vec4>>& emitters,
    int& max_depth,
    bool& show_window, 
    bool& show_fps, 
    bool& sub_window_process, 
    bool& capture,
    bool& gamma_corr,

    bool& camera_update,
    bool& scene_update,
    bool& renderer_update
) {
    // Begin the main menu bar at the top of the window
    if (ImGui::BeginMainMenuBar()) {
        // Create a menu item called "Options" in the main menu bar
        if (ImGui::BeginMenu("Options")) {
            // Add a checkbox in the menu to toggle the visibility of the collapsible window
            ImGui::MenuItem("Show Settings Window", NULL, &show_window);
            ImGui::MenuItem("Show Frame Rate Bar", NULL, &show_fps);
            ImGui::EndMenu(); // End the "Options" menu
        }
        if (ImGui::IsWindowFocused()) {
            sub_window_process = true;
        }
        ImGui::EndMainMenuBar(); // End the main menu bar
    }

    // Check if the collapsible window should be shown
    camera_update   = false;
    scene_update    = false;
    renderer_update = false;
    if (show_window) {
        // Begin the collapsible window
        if (ImGui::Begin("Settings", &show_window, ImGuiWindowFlags_AlwaysAutoResize)) {
            // Collapsible group for Camera Settings
            if (ImGui::CollapsingHeader("Camera Settings", ImGuiWindowFlags_AlwaysAutoResize)) {
                camera_update |= ImGui::Checkbox("orthogonal camera", &cam.use_orthogonal); // Toggles camera_bool_value on or off

                float value = focal2fov(cam.inv_focal, cam._hw);
                ImGui::Text("camera FoV");
                camera_update |= ImGui::SliderFloat("##slider", &value, 1.0f, 150.0f, "%.2f", ImGuiSliderFlags_None);
                ImGui::SameLine();
                ImGui::PushItemWidth(100.0f);
                camera_update |= ImGui::InputFloat("##input", &value, 1.0f, 150.0f, "%.2f");
                ImGui::PopItemWidth();
                cam.inv_focal = 1.f / fov2focal(value, cam._hw * 2.f);

                ImGui::Checkbox("Gamma Correction", &gamma_corr);
            }

            if (ImGui::CollapsingHeader("Scene Settings", ImGuiWindowFlags_AlwaysAutoResize)) {
                // Empty placeholder for future Scene Settings
                ImGui::Text("Scene emitter settings");
                for (auto& [name, e_val]: emitters) {
                    scene_update |= draw_rgb_ui(name, e_val.x(), e_val.y(), e_val.z(), e_val.w());
                }
            }

            if (ImGui::CollapsingHeader("Renderer Settings", ImGuiWindowFlags_AlwaysAutoResize)) {
                // Empty placeholder for future Renderer Settings
                ImGui::Text("Max bounces");
                ImGui::SameLine();
                ImGui::PushItemWidth(100.0f);
                renderer_update |= ImGui::InputInt("##max_depth", &max_depth, 1, 128); ImGui::SameLine();
                ImGui::PopItemWidth();
                ImGui::Separator();
            }
            if (ImGui::CollapsingHeader("Screen Capture", ImGuiWindowFlags_AlwaysAutoResize)) {
                capture = ImGui::Button("Capture Frame");
            }
            if (ImGui::IsWindowFocused()) {
                sub_window_process = true;
            }
            ImGui::End();
        }
    }
}

}   // namespace gui