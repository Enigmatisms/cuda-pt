/**
 * Online visualization based on imgui
 * There we have some utility functions
*/

#include <driver_types.h>
#include <GLFW/glfw3.h>
#include "core/max_depth.h"

class DeviceCamera;
class BSDFInfo;

namespace gui {

using gl_uint = unsigned int;

struct GUIParams {
    GUIParams() {
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
    bool renderer_update = false;
    bool serialized_update = false;
    bool output_png = true;

    std::vector<char> serialized_data;

    inline bool buffer_flush_update() const noexcept {
        return camera_update || scene_update || material_update || renderer_update || serialized_update;
    }

    inline void reset() {
        capture = false;
        camera_update   = false;
        scene_update    = false; 
        renderer_update = false,
        material_update = false;
        serialized_update = false;
    }
};

// initialize GL texture and PBO (pixel buffer object)
void init_texture_and_pbo(
    float* output_buffer,
    cudaGraphicsResource_t& pbo_resc, 
    gl_uint& pbo_id, gl_uint& cuda_texture_id,
    int width, int height
);

void update_texture(
    gl_uint pbo_id, gl_uint cuda_texture_id,
    int width, int height
);

void clean_up(
    GLFWwindow* window,
    uint32_t& pbo_id, uint32_t& cuda_texture_id
);

struct GLFWWindowDeleter {
    void operator() (GLFWwindow* window) const {
        if (window) {
            glfwDestroyWindow(window);
            std::cout << "GLFW window destroyed." << std::endl;
        }
    }
};

// draw images on the background of the main window
void window_render(
    gl_uint cuda_texture_id, 
    int width, int height
);

// window for displaying FPS & SPP
void show_render_statistics(
    int num_sample, 
    bool show_fps
);

// create a floating window (collapse-able) and draw image in it
void sub_window_render(std::string sub_win_name, gl_uint cuda_texture_id, int width, int height);

// create the main window
std::unique_ptr<GLFWwindow, GLFWWindowDeleter> create_window(int width, int height);

// process keyboard input and update camera position (host side)
bool keyboard_camera_update(DeviceCamera& camera, float step, bool& frame_cap, bool& exiting);

// process mouse input and update camera orientation (host side)
bool mouse_camera_update(DeviceCamera& cam, float sensitivity = 1);

// settings UI
void render_settings_interface(
    DeviceCamera& cam, 
    std::vector<std::pair<std::string, Vec4>>& emitters,
    std::vector<BSDFInfo>& bsdf_infos,
    MaxDepthParams& md_params,
    GUIParams& params,
    const uint8_t rdr_type = 0
);

}   // namespace gui