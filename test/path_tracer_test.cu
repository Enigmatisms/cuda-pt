#include "core/aos.cuh"
#include "core/bsdf.cuh"
#include "core/object.cuh"
#include "core/emitter.cuh"
#include "core/camera_model.cuh"
#include "core/virtual_funcs.cuh"
#include "renderer/path_tracer.cuh"
#include <ext/lodepng/lodepng.h>

__constant__ DeviceCamera dev_cam;
__constant__ Emitter* c_emitter[9];
__constant__ BSDF*    c_material[32];

int main(int argc, char** argv) {
    // right, down, back, left, up
    int num_triangle = 10, num_spheres = 3, num_prims = num_triangle + num_spheres;
    int num_material = 6, num_emitters = 1;
    int spp       = argc > 1 ? atoi(argv[1]) : 128;
    std::vector<Vec3> v1s = {{1, 1, 1}, {1, 1, 1}, {-1, 1, -1}, {-1, 1, -1}, {-1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {-1, 1, 1}, {-1,-1, 1}, {-1, -1, 1}, {0.5, 0, -0.7}, {-0.4,0.4, -0.5}, {-0.5, -0.5, -0.7}};
    std::vector<Vec3> v2s = {{1,-1,-1}, {1, -1,1}, {1, 1,  -1}, {1, -1, -1}, {1, 1,  1}, {1, 1, -1}, {-1, 1,  1}, {-1, 1,-1}, { 1,-1, 1}, {1,  1,  1}, {0.3, 0, 0}, {0.5, 0, 0}, {0.3, 0, 0}};
    std::vector<Vec3> v3s = {{1, 1,-1}, {1,-1,-1}, {1, -1, -1}, {-1, -1,-1}, {1, 1, -1}, {-1,1, -1}, {-1, -1,-1}, {-1,-1,-1}, { 1, 1, 1}, {-1, 1,  1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<Vec3> norms = {{-1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, -1, 0}, {0, -1, 0}, {1, 0, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0, -1}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}};

    Vec2 uv_default     = {0.0, 0.0};

    // scene setup
    ArrayType<Vec3> vert_data(v1s.size()), norm_data(v1s.size());
    ArrayType<Vec2> uvs_data(v1s.size());
    vert_data.from_vectors(v1s, v2s, v3s);
    norm_data.from_vectors(norms, norms, norms);
    uvs_data.fill(uv_default);

    std::vector<ObjInfo> objects;
    objects.emplace_back(0, 0, 2);
    objects.back().setup(vert_data);
    objects.emplace_back(1, 2, 2);
    objects.back().setup(vert_data);
    objects.emplace_back(1, 4, 2);
    objects.back().setup(vert_data);
    objects.emplace_back(2, 6, 2);
    objects.back().setup(vert_data);
    objects.emplace_back(1, 8, 2);
    objects.back().setup(vert_data);
    objects.emplace_back(3, 10, 1);
    objects.back().setup(vert_data, false);
    objects.emplace_back(4, 11, 1);
    objects.back().setup(vert_data, false);
    objects.emplace_back(5, 12, 1);
    objects.back().setup(vert_data, false);


    // TODO: this is not correct
    BSDF** pure_bsdfs;
    CUDA_CHECK_RETURN(cudaMalloc(&pure_bsdfs, sizeof(BSDF*) * num_material));
    create_bsdf<LambertianBSDF><<<1, 1>>>(pure_bsdfs, Vec4(1, 0.2, 0.2), Vec4(0, 0, 0), Vec4(0, 0, 0));     // red right
    create_bsdf<LambertianBSDF><<<1, 1>>>(pure_bsdfs + 1, Vec4(0.8, 0.8, 0.8), Vec4(0, 0, 0), Vec4(0, 0, 0));
    create_bsdf<LambertianBSDF><<<1, 1>>>(pure_bsdfs + 2, Vec4(0.2, 1, 0.2), Vec4(0, 0, 0), Vec4(0, 0, 0));
    create_bsdf<SpecularBSDF><<<1, 1>>>(pure_bsdfs + 3, Vec4(0, 0, 0), Vec4(0.9, 0.9, 0.9), Vec4(0, 0, 0));
    create_bsdf<TranslucentBSDF><<<1, 1>>>(pure_bsdfs + 4, Vec4(1.5, 0, 0), Vec4(0.99, 0.99, 0.99), Vec4(0, 0, 0));
    create_bsdf<LambertianBSDF><<<1, 1>>>(pure_bsdfs + 5, Vec4(0.99, 0.99, 0.99), Vec4(0, 0, 0), Vec4(0, 0, 0));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_material, pure_bsdfs, num_material * sizeof(BSDF*)));

    Emitter** pure_emitters; // null_emitter
    CUDA_CHECK_RETURN(cudaMalloc(&pure_emitters, sizeof(Emitter*) * (num_emitters + 1)));
    create_abstract_source<<<1, 1>>>(pure_emitters[0]);
    create_point_source<<<1, 1>>>(pure_emitters[1], Vec4(2, 2, 2), Vec3(0, 0, 0.8));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_emitter, pure_emitters, (num_emitters + 1) * sizeof(Emitter*)));
    
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
    
    PathTracer pt(objects, shapes, vert_data, norm_data, uvs_data, 1, width, height);
    printf("Prepare to render the scene...\n");
    auto bytes_buffer = pt.render(spp, 20);

    std::string file_name = "render.png";

    if (unsigned error = lodepng::encode(file_name, bytes_buffer, width, height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }

    printf("image saved to `%s`\n", file_name.c_str());

    vert_data.destroy();
    norm_data.destroy();
    uvs_data.destroy();

    destroy_gpu_alloc<<<1, num_material>>>(pure_bsdfs);
    destroy_gpu_alloc<<<1, 2>>>(pure_emitters);

    CUDA_CHECK_RETURN(cudaFree(pure_bsdfs));
    CUDA_CHECK_RETURN(cudaFree(pure_emitters));

    return 0;
}