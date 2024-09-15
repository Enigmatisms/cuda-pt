/**
 * Online visualization based on imgui
 * There we have some utility functions
*/

#include <driver_types.h>

namespace gui {

// initialize GL texture and PBO (pixel buffer object)
void init_texture_and_pbo(
    float* output_buffer,
    cudaGraphicsResource_t& pbo_resc, 
    GLuint& pbo_id, GLuint& cuda_texture_id,
    int width, int height
);

void window_render(GLuint cuda_texture_id, int width, int height);

void update_texture(
    GLuint pbo_id, GLuint cuda_texture_id,
    int width, int height
);

void clean_up(
    GLFWwindow* window,
    GLuint& pbo_id, GLuint& cuda_texture_id
);

struct GLFWWindowDeleter {
    void operator() (GLFWwindow* window) const {
        if (window) {
            glfwDestroyWindow(window);
            std::cout << "GLFW window destroyed." << std::endl;
        }
    }
};

std::unique_ptr<GLFWwindow, GLFWWindowDeleter> create_window(int width, int height);

}   // namespace gui