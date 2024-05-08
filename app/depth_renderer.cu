#include "core/soa.cuh"
#include "renderer/depth.cuh"
#include "core/camera_model.cuh"

__constant__ DeviceCamera dev_cam;

int main() {
    // down, back, right, left, up
    int num_prims = 10;
    std::vector<Vec3> v1s = {{1, -1,  0}, {1, -1, 0}, {1, 1, -1}, {1, 1,  -1}, {1, -1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}, {-1, 1,  1}, {1, 1, 1}};
    std::vector<Vec3> v2s = {{-1, 1,  0}, {1, 1,  0}, {1, 1,  1}, {-1, 1,  1}, {1, -1,  1}, {1,  1,  1}, {-1,  1, -1}, {-1, 1,  1}, {1,  1,  1}, {1, -1, 1}};
    std::vector<Vec3> v3s = {{-1, -1, 0}, {-1, 1, 0}, {-1, 1, 1}, {-1, 1, -1}, {1,  1,  1}, {1,  1, -1}, {-1, -1,  1}, {-1, -1, 1}, {-1, -1, 1}, {-1, -1, 1}};
    Vec3 normal_default = {0, 1, 0};
    Vec2 uv_default     = {0.5, 0.5};

    // scene setup
    SoA3<Vec3> vert_data(v1s.size()), norm_data(v1s.size());
    SoA3<Vec2> uvs_data(v1s.size());
    vert_data.from_vectors(v1s, v2s, v3s);
    norm_data.fill(normal_default);
    uvs_data.fill(uv_default);

    // camera setup
    Vec3 from(0, 0, -3), to(0, 0, 0);
    int width = 800, height = 800;
    float fov = 90;
    DeviceCamera camera(from, to, fov, width, height);
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(dev_cam, &camera, sizeof(DeviceCamera)));

    // shape setup
    Shape* shapes;
    CUDA_CHECK_RETURN(cudaMallocManaged(&shapes, sizeof(Shape) * 10));
    for (int i = 0; i < 5; i++)
        shapes[i] = TriangleShape(i >> 1);
    
    // aabb setup
    AABB* aabbs;
    CUDA_CHECK_RETURN(cudaMallocManaged(&aabbs, sizeof(AABB) * 10));
    for (int i = 0; i < 10; i++)
        aabbs[i] = AABB(v1s[i], v2s[i], v3s[i]);

    auto bytes_buffer = render_depth(shapes, aabbs, vert_data, norm_data, uvs_data, dev_cam, num_prims, width, height);

    CUDA_CHECK_RETURN(cudaFree(shapes));
    CUDA_CHECK_RETURN(cudaFree(aabbs));
    return 0;
}