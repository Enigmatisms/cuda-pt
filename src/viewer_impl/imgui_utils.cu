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
#include "core/camera_model.cuh"
#include "core/cuda_utils.cuh"
#include "core/dynamic_bsdf.cuh"
#include "core/enums.cuh"
#include "core/imgui_utils.cuh"
#include "core/serialize.h"
#include <cuda_gl_interop.h>
#include <ext/imgui/backends/imgui_impl_glfw.h>
#include <ext/imgui/backends/imgui_impl_opengl3.h>
#include <ext/imgui/imgui.h>

namespace gui {

static constexpr const char *RENDERER_NAMES[] = {
    "Megakernel Path Tracing",  "Wavefront Path Tracing",
    "Megakernel Light Tracing", "Voxel-SDF Path Tracing",
    "Scene Depth Tracing",      "BVH Cost Visualizer"};

static constexpr std::array<const char *, 4> COLOR_MAP_NAMES = {
    "Jet", "Plasma", "Viridis", "GrayScale"};

static void setup_imgui_style(bool dark_style, float alpha_) {
    ImGuiStyle &style = ImGui::GetStyle();

    // style settings from
    // https://gist.github.com/dougbinks/8089b4bbaccaaf6fa204236978d165a9
    style.Alpha = 1.0f;
    style.FrameRounding = 3.f;
    style.WindowRounding = 3.f;
    style.TabRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.ChildRounding = 3.0f;
    style.PopupRounding = 3.0f;
    style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
    style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
    style.Colors[ImGuiCol_TitleBgCollapsed] =
        ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
    style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered] =
        ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
    style.Colors[ImGuiCol_ScrollbarGrabActive] =
        ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
    style.Colors[ImGuiCol_SliderGrabActive] =
        ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
    style.Colors[ImGuiCol_ResizeGripHovered] =
        ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
    style.Colors[ImGuiCol_ResizeGripActive] =
        ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
    style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
    style.Colors[ImGuiCol_PlotLinesHovered] =
        ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    style.Colors[ImGuiCol_PlotHistogramHovered] =
        ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);

    if (dark_style) {
        for (int i = 0; i <= ImGuiCol_COUNT; i++) {
            ImVec4 &col = style.Colors[i];
            float H, S, V;
            ImGui::ColorConvertRGBtoHSV(col.x, col.y, col.z, H, S, V);

            if (S < 0.1f)
                V = 1.0f - V;
            ImGui::ColorConvertHSVtoRGB(H, S, V, col.x, col.y, col.z);
            if (col.w < 1.00f)
                col.w *= alpha_;
        }
    } else {
        for (int i = 0; i < ImGuiCol_COUNT; i++) {
            ImVec4 &col = style.Colors[i];
            if (col.w < 1.00f) {
                col.x *= alpha_;
                col.y *= alpha_;
                col.z *= alpha_;
                col.w *= alpha_;
            }
        }
    }

    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("../../assets/fonts/nunito-sans.ttf", 16);
}

// initialize GL texture and PBO (pixel buffer object)
void init_texture_and_pbo(cudaGraphicsResource_t &pbo_resc, gl_uint &pbo_id,
                          gl_uint &cuda_texture_id, int width, int height) {
    // create PBO
    glGenBuffers(1, &pbo_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(float), 0,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register PBO
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(
        &pbo_resc, pbo_id, cudaGraphicsMapFlagsWriteDiscard));

    // Create OpenGL texture (context)
    glGenTextures(1, &cuda_texture_id);
    glBindTexture(GL_TEXTURE_2D, cuda_texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
                 GL_FLOAT, NULL);
}

void sub_window_render(std::string sub_win_name, gl_uint cuda_texture_id,
                       int width, int height) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin(sub_win_name.c_str());
    ImVec2 wh(width, height);
    ImGui::Image(cuda_texture_id, wh);
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void window_render(GLuint cuda_texture_id, int width, int height) {
    ImDrawList *draw_list = ImGui::GetBackgroundDrawList();
    ImVec2 top_left(0, 0);
    ImVec2 bottom_right(width, height);
    // splat pixel buffer outputs
    draw_list->AddImage(cuda_texture_id, top_left, bottom_right);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void show_render_statistics(int num_sample, bool show_fps) {
    if (show_fps) {
        const char *window_title = "Statistics";
        ImVec2 text_size = ImGui::CalcTextSize(window_title);
        float padding = ImGui::GetStyle().WindowPadding.x * 2;
        float min_width = text_size.x * 1.2f + padding;
        ImGui::SetNextWindowSizeConstraints(ImVec2(min_width, 0),
                                            ImVec2(FLT_MAX, FLT_MAX));
        ImGui::Begin(window_title, &show_fps);
        ImGuiIO &io = ImGui::GetIO();
        ImGui::Text("FPS: %.2f", io.Framerate);
        ImGui::Text("SPP: %4d", num_sample);
        ImGui::End();
    }
}

void update_texture(gl_uint pbo_id, gl_uint cuda_texture_id, int width,
                    int height) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
    glBindTexture(GL_TEXTURE_2D, cuda_texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT,
                    NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void clean_up(GLFWwindow *window, gl_uint &pbo_id, gl_uint &cuda_texture_id) {
    glDeleteBuffers(1, &pbo_id);
    glDeleteTextures(1, &cuda_texture_id);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

std::unique_ptr<GLFWwindow, GLFWWindowDeleter> create_window(int width,
                                                             int height) {
    // initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }

    // OPENGL window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(width, height, "CUDA-PT", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    // vertical sync
    glfwSwapInterval(1);

    // initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return nullptr;
    }

    // initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    setup_imgui_style(true, 0.75);
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    return std::unique_ptr<GLFWwindow, GLFWWindowDeleter>(window);
}

bool keyboard_camera_update(DeviceCamera &camera, float step, bool &frame_cap,
                            bool &exiting) {
    bool update = false;
    if (ImGui::IsKeyDown(ImGuiKey_W)) {
        update = true;
        camera.move_forward(step);
    }
    if (ImGui::IsKeyDown(ImGuiKey_S)) {
        update = true;
        camera.move_backward(step);
    }
    if (ImGui::IsKeyDown(ImGuiKey_A)) {
        update = true;
        camera.move_left(step);
    }
    if (ImGui::IsKeyDown(ImGuiKey_D)) {
        update = true;
        camera.move_right(step);
    }
    if (ImGui::IsKeyDown(ImGuiKey_P)) {
        frame_cap = true;
    }
    if (ImGui::IsKeyDown(ImGuiKey_Escape)) {
        exiting = true;
    }
    if (ImGui::IsKeyDown(ImGuiKey_E)) {
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

bool mouse_camera_update(DeviceCamera &cam, float sensitivity) {
    ImGuiIO &io = ImGui::GetIO();
    // Left button is pressed?
    if (io.MouseDown[0]) {
        // get current mouse pos

        // whether the mouse is being dragged
        if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) {
            float deltaX = io.MouseDelta.x;
            float deltaY = io.MouseDelta.y;

            float yaw = deltaX * sensitivity * DEG2RAD;
            float pitch = deltaY * sensitivity * DEG2RAD;

            cam.rotate(yaw, pitch);
            return true;
        }
    }
    return false;
}

static bool draw_color_picker(std::string label, std::string name,
                              float *color_start) {
    // squared box
    ImGui::Text(name.c_str());
    ImGui::SameLine();
    ImGuiColorEditFlags color_edit_flags =
        ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoAlpha |
        ImGuiColorEditFlags_InputRGB | ImGuiColorEditFlags_NoLabel |
        ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_PickerHueBar;
    label = "##rgb-" + label;
    return ImGui::ColorEdit3(label.c_str(), color_start, color_edit_flags);
}

static bool draw_coupled_slider_input(std::string id, std::string name,
                                      float &val, float min_val = 0.0,
                                      float max_val = 1.f) {
    bool updated = false;
    ImGui::Text(name.c_str());
    ImGui::SameLine();

    ImGui::PushItemWidth(120.0f);
    std::string label = "##ScalerSlider-" + id;
    updated |=
        ImGui::SliderFloat(label.c_str(), &val, min_val, max_val, "%.3f");
    ImGui::SameLine();
    label = "##ScalerInput-" + id;
    updated |= ImGui::InputFloat(label.c_str(), &val, min_val, max_val, "%.3f");
    ImGui::PopItemWidth();
    return updated;
}

static bool draw_customized_check_box(std::string id, std::string name,
                                      bool &val) {
    ImGui::Text(name.c_str());
    ImGui::SameLine();
    std::string label = "##select-" + id;
    return ImGui::Checkbox(label.c_str(), &val);
}

static bool draw_integer_input(std::string id, std::string name, int &val,
                               int min_val = 1, int max_val = 64,
                               float width = 100.f) {
    bool updated = false;
    ImGui::Text(name.c_str());
    ImGui::SameLine();
    ImGui::PushItemWidth(width);
    updated |= ImGui::InputInt(("##" + id).c_str(), &val, min_val, max_val);
    ImGui::PopItemWidth();
    ImGui::Separator();
    return updated;
}

template <int N>
static bool draw_selection_menu(const std::array<const char *, N> &picks,
                                std::string id, std::string name,
                                uint8_t &current_value) {
    ImGui::Text("%s", name);
    ImGui::SameLine();

    const char *current_name = picks[current_value];

    bool changed = false;
    if (ImGui::BeginCombo(id.c_str(), current_name)) {
        for (uint8_t i = 0; i < N; ++i) {
            bool is_selected = (current_value == i);
            if (ImGui::Selectable(picks[i], is_selected)) {
                current_value = i;
                changed = !is_selected;
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
        return changed;
    }
    return changed;
}

static bool emitter_widget(std::string description, Vec4 &c_scaler,
                           bool add_rule = true) {
    bool updated = false;
    if (add_rule)
        ImGui::Separator();
    if (c_scaler.x() < 0) {
        ImGui::Text("Environment Map '%s'", description.c_str());
        updated |= draw_coupled_slider_input(description + "-scale", "Scaler",
                                             c_scaler.y(), 0.f, 100.f);
        updated |= draw_coupled_slider_input(
            description + "-azimuth", "Azimuth", c_scaler.z(), -180.f, 180.f);
        updated |= draw_coupled_slider_input(description + "-zenith", "Zenith",
                                             c_scaler.w(), 0.f, 180.f);
    } else {
        ImGui::Text("Emission for '%s'", description.c_str());
        updated |= draw_color_picker(description, "Color", &c_scaler.x());
        updated |= draw_coupled_slider_input(description, "Scaler",
                                             c_scaler.w(), 0.f, 100.f);
    }
    return updated;
}

static bool material_widget(std::vector<BSDFInfo> &bsdf_infos) {
    bool updated = false;
    for (auto &info : bsdf_infos) {
        ImGui::Separator();
        std::string header_name = "Material '" + info.name + "' | Type: '" +
                                  BSDF_NAMES[info.type] + "'";
        ImGui::Indent();
        if (info.in_use == false) {
            ImGui::PushStyleColor(ImGuiCol_Header,
                                  ImVec4(0.4f, 0.4f, 0.4f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                                  ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive,
                                  ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(0.8f, 0.8f, 0.8f, 0.8f));
            header_name = "[NOT USED] Material '" + info.name + "'";
            if (ImGui::CollapsingHeader(header_name.c_str(),
                                        ImGuiWindowFlags_AlwaysAutoResize))
                ImGui::Text("This BSDF of type '%s' is not used.",
                            BSDF_NAMES[info.type]);
            ImGui::PopStyleColor(4);
            ImGui::Unindent();
            continue;
        }
        if (ImGui::CollapsingHeader(header_name.c_str(),
                                    ImGuiWindowFlags_AlwaysAutoResize)) {
            bool local_update = draw_selection_menu<BSDFType::NumSupportedBSDF>(
                BSDF_NAMES, "##" + info.name + "-mat", "BSDF",
                reinterpret_cast<uint8_t &>(info.type));
            info.bsdf_changed = local_update;
            if (info.bsdf_changed) {
                updated = true;
            }
            if (local_update) {
                ImGui::Unindent();
                continue;
            }
            if (info.type == BSDFType::GGXConductor) {
                local_update |= draw_color_picker(info.name + "-kd", "Albedo",
                                                  &info.bsdf.k_g.x());
                local_update |= draw_selection_menu<MetalType::NumMetalType>(
                    METAL_NAMES, "##" + info.name + "-type", "Metal Type",
                    info.bsdf.mtype);
                local_update |= draw_coupled_slider_input(
                    info.name + "rx", "roughness X", info.bsdf.roughness_x());
                local_update |= draw_coupled_slider_input(
                    info.name + "ry", "roughness Y", info.bsdf.roughness_y());
            } else if (info.type == BSDFType::Plastic ||
                       info.type == BSDFType::PlasticForward) {
                local_update |= draw_color_picker(
                    info.name + "-kd", "Substrate", &info.bsdf.k_d.x());
                updated |= draw_color_picker(info.name + "-ks", "Coating",
                                             &info.bsdf.k_s.x());
                local_update |= draw_color_picker(
                    info.name + "-kg", "Absorption", &info.bsdf.k_g.x());
                local_update |= draw_coupled_slider_input(
                    info.name + "-ior", "IoR", info.bsdf.ior(), 1.0, 3.0);
                local_update |= draw_coupled_slider_input(
                    info.name + "-thc", "Thickness", info.bsdf.thickness());
                local_update |= draw_coupled_slider_input(
                    info.name + "-trp", "Transmit Proba",
                    info.bsdf.trans_scaler());
                bool is_selected = info.bsdf.penetration() > 0;
                if (draw_customized_check_box(info.name, "Penetrable",
                                              is_selected)) {
                    info.bsdf.penetration() = is_selected;
                    local_update = true;
                }
            } else if (info.type == BSDFType::Translucent) {
                local_update |= draw_color_picker(info.name + "-ks", "Specular",
                                                  &info.bsdf.k_s.x());
                local_update |= draw_coupled_slider_input(
                    info.name + "ior", "IoR", info.bsdf.k_d.x());
            } else if (info.type == BSDFType::Lambertian) {
                local_update |= draw_color_picker(info.name + "-kd", "Diffuse",
                                                  &info.bsdf.k_d.x());
            } else if (info.type == BSDFType::Specular) {
                local_update |= draw_color_picker(info.name + "-ks", "Specular",
                                                  &info.bsdf.k_s.x());
            } else if (info.type == BSDFType::Dispersion) {
                local_update |= draw_color_picker(info.name + "-ks", "Specular",
                                                  &info.bsdf.k_s.x());
                local_update |=
                    draw_selection_menu<DispersionType::NumDispersionType>(
                        DISPERSION_NAMES, "##" + info.name + "-type",
                        "Dispersion Type", info.bsdf.mtype);
            } else if (info.type == BSDFType::Forward) {
                ImGui::Text("Forward BSDF has nothing adjustable.");
            }
            info.updated = local_update;
            updated |= local_update;
        }
        ImGui::Unindent();
    }
    return updated;
}

static bool medium_widget(std::vector<MediumInfo> &med_infos) {
    bool updated = false;
    for (size_t i = 1; i < med_infos.size(); i++) {
        auto &info = med_infos[i];
        ImGui::Separator();
        std::string header_name = "Medium '" + info.name + "' | Type: '" +
                                  MEDIUM_NAMES[info.mtype] + "'";
        ImGui::Indent();
        if (ImGui::CollapsingHeader(header_name.c_str(),
                                    ImGuiWindowFlags_AlwaysAutoResize)) {
            bool local_update =
                draw_selection_menu<PhaseFuncType::NumSupportedPhase>(
                    PHASES_NAMES, "##" + info.name + "-ptype", "Phase",
                    reinterpret_cast<uint8_t &>(info.ptype));
            info.phase_changed = local_update;
            if (info.phase_changed) {
                updated = true;
            }

            if (local_update) {
                ImGui::Unindent();
                continue;
            }
            {
                ImGui::Indent();
                std::string header_name = "Phase '" + info.name +
                                          "' | Type: '" +
                                          PHASES_NAMES[info.ptype] + "'";
                if (ImGui::CollapsingHeader(
                        header_name.c_str(),
                        ImGuiWindowFlags_AlwaysAutoResize)) {
                    if (info.ptype == PhaseFuncType::HenyeyGreenstein) {
                        local_update |= draw_coupled_slider_input(
                            info.name + "-g", "g", info.med_param.g(), -0.999,
                            0.999);
                    } else if (info.ptype == PhaseFuncType::DuoHG) {
                        local_update |= draw_coupled_slider_input(
                            info.name + "-g1", "g(1)", info.med_param.g1(),
                            -0.999, 0.999);
                        local_update |= draw_coupled_slider_input(
                            info.name + "-g2", "g(2)", info.med_param.g2(),
                            -0.999, 0.999);
                        local_update |= draw_coupled_slider_input(
                            info.name + "-weight", "Weight",
                            info.med_param.weight());
                    } else {
                        ImGui::Text("No modifi-able params.");
                    }
                }
                ImGui::Unindent();
            }
            if (info.mtype == MediumType::Homogeneous) {
                local_update |=
                    draw_color_picker(info.name + "-sigma_a", "Absorption",
                                      &info.med_param.sigma_a.x());
                local_update |=
                    draw_color_picker(info.name + "-sigma_s", "Scattering",
                                      &info.med_param.sigma_s.x());
                local_update |=
                    draw_coupled_slider_input(info.name + "-scale", "Scale",
                                              info.med_param.scale, 0.1, 100.f);
            } else if (info.mtype == MediumType::Grid) {
                local_update |=
                    draw_color_picker(info.name + "-albedo", "Albedo",
                                      &info.med_param.sigma_a.x());
                local_update |=
                    draw_coupled_slider_input(info.name + "-scale", "Scale",
                                              info.med_param.scale, 0.1, 100.f);
                local_update |= draw_coupled_slider_input(
                    info.name + "-tp-scale", "Temp Scale",
                    info.med_param.temperature_scale(), 0.1f, 10.f);
                local_update |= draw_coupled_slider_input(
                    info.name + "-em-scale", "Emission Scale",
                    info.med_param.emission_scale(), 0.1f, 100.f);
            }

            info.updated = local_update;
            updated |= local_update;
        }
        ImGui::Unindent();
    }
    return updated;
}

void render_settings_interface(
    DeviceCamera &cam, std::vector<std::pair<std::string, Vec4>> &emitters,
    std::vector<BSDFInfo> &bsdf_infos, std::vector<MediumInfo> &med_infos,
    MaxDepthParams &md_params, GUIParams &params, const uint8_t rdr_type) {
    // Begin the main menu bar at the top of the window
    if (ImGui::BeginMainMenuBar()) {
        // Create a menu item called "Options" in the main menu bar
        if (ImGui::BeginMenu("Options")) {
            // Add a checkbox in the menu to toggle the visibility of the
            // collapsible window
            ImGui::MenuItem("Show Settings Window", NULL, &params.show_window);
            ImGui::MenuItem("Show Frame Rate Bar", NULL, &params.show_fps);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // Check if the collapsible window should be shown
    params.scene_update = false;
    params.renderer_update = false;
    params.material_update = false;
    params.medium_update = false;
    if (params.show_window) {
        // Begin the collapsible window
        if (ImGui::Begin("Settings", &params.show_window,
                         ImGuiWindowFlags_AlwaysAutoResize)) {
            // Collapsible group for Camera Settings
            if (ImGui::CollapsingHeader("Camera Settings",
                                        ImGuiWindowFlags_AlwaysAutoResize)) {
                params.camera_update |= ImGui::Checkbox(
                    "orthogonal camera",
                    &cam.use_orthogonal); // Toggles camera_bool_value on or off

                float value = focal2fov(cam.inv_focal, cam._hw);
                params.camera_update |= draw_coupled_slider_input(
                    "Fov", "Camera FoV", value, 1.0f, 150.f);
                cam.inv_focal = 1.f / fov2focal(value, cam._hw * 2.f);

                ImGui::Checkbox("Gamma Correction", &params.gamma_corr);
                draw_coupled_slider_input("cam-speed", "Camera Speed",
                                          params.trans_speed, 0.01f, 2.f);
                draw_coupled_slider_input("rot-speed", "Rotation Speed",
                                          params.rot_sensitivity, 0.1f, 2.f);
            }

            if (ImGui::CollapsingHeader("Scene Settings",
                                        ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("Scene emitter settings");
                for (auto &[name, e_val] : emitters) {
                    params.scene_update |= emitter_widget(name, e_val);
                }
            }

            if (ImGui::CollapsingHeader("Renderer Settings",
                                        ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text(RENDERER_NAMES[rdr_type]);
                ImGui::Separator();
                params.renderer_update |= draw_integer_input(
                    "max_depth", "Max Depth", md_params.max_depth);
                params.renderer_update |= draw_integer_input(
                    "max_diff", "Max Diffuse", md_params.max_diffuse);
                params.renderer_update |= draw_integer_input(
                    "max_spec", "Max Specular", md_params.max_specular);
                params.renderer_update |= draw_integer_input(
                    "max_trans", "Max Transmit", md_params.max_tranmit);
                if (rdr_type == RendererType::MegaKernelVPT) {
                    params.renderer_update |= draw_integer_input(
                        "max_vol", "Max Volume", md_params.max_volume);
                    params.renderer_update |= draw_coupled_slider_input(
                        "min_time", "Min Time", md_params.min_time, 0,
                        md_params.max_time);
                    params.renderer_update |= draw_coupled_slider_input(
                        "max_time", "Max Time", md_params.max_time, 0, 20);
                }
                ImGui::Checkbox("Output PNG", &params.output_png);
                if (!params.output_png) {
                    ImGui::PushItemWidth(120.0f);
                    ImGui::InputInt("Compression Quality", &params.compress_q,
                                    1, 100);
                    ImGui::PopItemWidth();
                }
                if (rdr_type == RendererType::DepthTracing ||
                    rdr_type == RendererType::BVHCostViz) {
                    uint8_t update_v =
                        Serializer::get<int>(params.serialized_data, 0) & 0x7f;
                    if (ImGui::Checkbox("Log2 Transform",
                                        &params.log2_output) ||
                        draw_selection_menu<4>(COLOR_MAP_NAMES, "##color-map",
                                               "Color Map", update_v)) {
                        int data =
                            params.log2_output ? (update_v | 0x80) : update_v;
                        Serializer::set<int>(params.serialized_data, 0, data);
                        params.serialized_update = true;
                    }
                    if (rdr_type == RendererType::BVHCostViz) {
                        int max_query = ceilf(
                            Serializer::get<float>(params.serialized_data, 2));
                        ImGui::Text(("Max value: " + std::to_string(max_query))
                                        .c_str());

                        int max_value =
                            Serializer::get<int>(params.serialized_data, 1);
                        max_value = max_value > 0 ? max_value : max_query;
                        if (draw_integer_input("bvh-max-cost", "(Viz) Max Cost",
                                               max_value)) {
                            Serializer::set<int>(params.serialized_data, 1,
                                                 max_value);
                            params.serialized_update = true;
                        }
                    }
                }
            }
            if (ImGui::CollapsingHeader("Material Settings",
                                        ImGuiWindowFlags_AlwaysAutoResize)) {
                params.material_update |= material_widget(bsdf_infos);
            }
            if (ImGui::CollapsingHeader("Medium Settings",
                                        ImGuiWindowFlags_AlwaysAutoResize)) {
                params.medium_update |= medium_widget(med_infos);
            }
            if (ImGui::CollapsingHeader("Screen Capture",
                                        ImGuiWindowFlags_AlwaysAutoResize)) {
                params.capture = ImGui::Button("Capture Frame");
            }
        }
        ImGui::End();
    }
}
} // namespace gui
