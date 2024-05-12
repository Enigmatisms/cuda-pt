#include "core/soa.cuh"
#include "core/camera_model.cuh"
#include "renderer/depth.cuh"
#include <ext/lodepng/lodepng.h>

__constant__ DeviceCamera dev_cam;

int main() {
    InitProfiler();

    // right, down, back, left, up
    int num_triangle = 10, num_spheres = 3, num_prims = num_triangle + num_spheres;
    int spp       = 2;
    std::vector<Vec3> v1s = {{1, 1, 1}, {1, 1, 1}, {-1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {-1, 1, 1}, {-1,-1, 1}, {-1, -1, 1}, {0.5, 0, -0.7}, {-0.4,0.4, -0.5}, {-0.5, -0.5, -0.7}};
    std::vector<Vec3> v2s = {{1,-1,-1}, {1, -1,1}, {1, 1,  -1}, {1, -1, -1}, {1, 1,  1}, {1, 1, -1}, {-1, 1,  1}, {-1, 1,-1}, { 1,-1, 1}, {1,  1,  1}, {0.3, 0, 0}, {0.5, 0, 0}, {0.3, 0, 0}};
    std::vector<Vec3> v3s = {{1, 1,-1}, {1,-1,-1}, {1, -1, -1}, {-1, -1,-1}, {1, 1, -1}, {-1,1, -1}, {-1, -1,-1}, {-1,-1,-1}, { 1, 1, 1}, {-1, 1,  1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Vec3 normal_default = {0, 1, 0};
    Vec2 uv_default     = {0.5, 0.5};

    // scene setup
    SoA3<Vec3> vert_data(v1s.size()), norm_data(v1s.size());
    SoA3<Vec2> uvs_data(v1s.size());
    vert_data.from_vectors(v1s, v2s, v3s);
    norm_data.fill(normal_default);
    uvs_data.fill(uv_default);

    // camera setup
    Vec3 from(0, -3, 0), to(0, 0, 0);
    int width = 1024, height = 1024;
    float fov = 55;
    DeviceCamera camera(from, to, fov, width, height);
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(dev_cam, &camera, sizeof(DeviceCamera)));

    // shape setup
    std::vector<Shape> shapes(num_prims);
    for (int i = 0; i < num_triangle; i++)
        shapes[i] = TriangleShape(i >> 1);
    for (int i = num_triangle; i < num_prims; i++)
        shapes[i] = SphereShape(i >> 1);
    
    DepthTracer dtracer(shapes, vert_data, norm_data, uvs_data, width, height);
    auto bytes_buffer = dtracer.render(spp);

    std::string file_name = "depth-render.png";

    if (unsigned error = lodepng::encode(file_name, bytes_buffer, width, height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }

    printf("image saved to `%s`\n", file_name.c_str());

    vert_data.destroy();
    norm_data.destroy();
    uvs_data.destroy();

    // ReportThreadStats();    
    // PrintStats(stdout);
    ReportProfilerResults(stdout);

    // ClearStats();
    ClearProfiler();
    CleanupProfiler();
    return 0;
}