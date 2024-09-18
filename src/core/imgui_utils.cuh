/**
 * Online visualization based on imgui
 * There we have some utility functions
*/

#include <driver_types.h>
#include <GLFW/glfw3.h>

namespace gui {

using gl_uint = unsigned int;

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
    bool& sub_window_proc,
    bool show_fps
);

// create a floating window (collapse-able) and draw image in it
void sub_window_render(std::string sub_win_name, gl_uint cuda_texture_id, int width, int height);

// create the main window
std::unique_ptr<GLFWwindow, GLFWWindowDeleter> create_window(int width, int height);

// process keyboard input and update camera position (host side)
bool keyboard_camera_update(DeviceCamera& camera, float step = 0.1);

// process mouse input and update camera orientation (host side)
bool mouse_camera_update(DeviceCamera& cam, float sensitivity = 1);

// settings UI
bool render_settings_interface(DeviceCamera& cam, bool& show_window, bool& show_fps, bool& render_settings_interface, bool& capture);

}   // namespace gui