#include "core/soa.cuh"

int main() {
    // down, back, right, left, up
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

    // image buffer setup
    int width = 800, height = 800;
    DeviceImage image(width, height);
    
    // 


    return 0;
}