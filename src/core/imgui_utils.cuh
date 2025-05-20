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

/**
 * @author Qianyue He
 * @brief Online visualization based on imgui
 * There we have some utility functions
 * @date Unknown
 */

#include "core/max_depth.h"
#include "core/vec4.cuh"
#include <GLFW/glfw3.h>
#include <driver_types.h>
#include <memory>
#include <string>
#include <vector>

class DeviceCamera;
class BSDFInfo;
class MediumInfo;

namespace gui {

using gl_uint = unsigned int;

struct GUIParams {
    GUIParams(std::vector<char> &_data) : serialized_data(_data) {
        serialized_data.reserve(16);
    }

    int compress_q = 90;

    float trans_speed = 0.1f;
    float rot_sensitivity = 0.5f;

    bool show_window = true;
    bool show_fps = true;
    bool capture = false;
    bool gamma_corr = true;

    bool camera_update = false;
    bool scene_update = false;
    bool material_update = false;
    bool medium_update = false;
    bool renderer_update = false;
    bool serialized_update = false;
    bool output_png = true;
    bool log2_output = false;

    std::vector<char> &serialized_data;

    inline bool buffer_flush_update() const noexcept {
        return camera_update || scene_update || material_update ||
               medium_update || renderer_update;
    }

    inline void reset() {
        capture = false;
        camera_update = false;
        scene_update = false;
        medium_update = false;
        renderer_update = false, material_update = false;
        serialized_update = false;
    }
};

// initialize GL texture and PBO (pixel buffer object)
void init_texture_and_pbo(cudaGraphicsResource_t &pbo_resc, gl_uint &pbo_id,
                          gl_uint &cuda_texture_id, int width, int height);

void update_texture(gl_uint pbo_id, gl_uint cuda_texture_id, int width,
                    int height);

void clean_up(GLFWwindow *window, uint32_t &pbo_id, uint32_t &cuda_texture_id);

struct GLFWWindowDeleter {
    void operator()(GLFWwindow *window) const {
        if (window) {
            glfwDestroyWindow(window);
        }
    }
};

// draw images on the background of the main window
void window_render(gl_uint cuda_texture_id, int width, int height);

// window for displaying FPS & SPP
void show_render_statistics(int num_sample, bool show_fps);

// create a floating window (collapse-able) and draw image in it
void sub_window_render(std::string sub_win_name, gl_uint cuda_texture_id,
                       int width, int height);

// create the main window
std::unique_ptr<GLFWwindow, GLFWWindowDeleter> create_window(int width,
                                                             int height);

// process keyboard input and update camera position (host side)
bool keyboard_camera_update(DeviceCamera &camera, float step, bool &frame_cap,
                            bool &exiting);

// process mouse input and update camera orientation (host side)
bool mouse_camera_update(DeviceCamera &cam, float sensitivity = 1);

// settings UI
void render_settings_interface(
    DeviceCamera &cam, std::vector<std::pair<std::string, Vec4>> &emitters,
    std::vector<BSDFInfo> &bsdf_infos, std::vector<MediumInfo> &med_infos,
    MaxDepthParams &md_params, GUIParams &params, const uint8_t rdr_type = 0);

} // namespace gui
