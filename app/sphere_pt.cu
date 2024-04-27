#include <iostream>
#include <chrono>
#include <iomanip>
#include <curand_kernel.h>

#include "core/cuda_utils.cuh"
#include "core/ray.cuh"
#include "core/vec3.cuh"
#include "useless/sphere_base.cuh"
#include "ext/lodepng/lodepng.h"

CPT_CPU_GPU
inline float clamp(float x, float low, float high) {
    return x < low ? low : x > high ? high : x;
}

inline int toInt(float x) { return int(powf(clamp(x, 0, 1), 1 / 2.2) * 255 + .5); }

struct Sampler {
    CPT_GPU Sampler(int seed) {
        curand_init(seed, 0, 0, &rand_state);

    }

    CPT_GPU float generate() { return curand_uniform(&rand_state); }

private:
    curandState rand_state;
};

CPT_GPU
inline int intersect(const Ray &r, float &t, const Sphere *spheres, int num_spheres) {
    int id = -1;
    t = std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_spheres; ++i) {
        float d = spheres[i].intersect(r);
        if (d > EPSILON && d < t) {
            t = d;
            id = i;
        }
    }

    return id;
}

CPT_GPU
Vec3f trace(const Ray &camera_ray, const Sphere *spheres, const int num_spheres, Sampler &sampler) {
    Vec3f radiance(0.0, 0.0, 0.0);
    Vec3f throughput(1.0, 1.0, 1.0);

    auto ray = camera_ray;

    for (int depth = 0;; ++depth) {
        float t; // distance to intersection
        int hit_sphere_id = intersect(ray, t, spheres, num_spheres);
        if (hit_sphere_id < 0) {
            break;
        }

        const Sphere &obj = spheres[hit_sphere_id]; // the hit object
        Vec3f hit_point = ray.o + ray.d * t;
        Vec3f surface_normal = (hit_point - obj.position).normalized(); // always face out
        Vec3f normal = surface_normal.dot(ray.d) < 0 ? surface_normal : surface_normal * -1;

        radiance += throughput * obj.emission;
        throughput *= obj.color;

        if (depth > 4) {
            // russian roulette
            float probability_russian_roulette = clamp(throughput.max_elem(), 0.1, 0.95);

            if (sampler.generate() >= probability_russian_roulette) {
                // terminated
                break;
            }
            // survive and enhanced
            throughput *= (1.0f / probability_russian_roulette);
        }

        if (obj.reflection_type == ReflectionType::DIFFUSE) { // Ideal DIFFUSE reflection
            float r1 = 2 * M_PI * sampler.generate();
            float r2 = sampler.generate();
            float r2s = sqrtf(r2);
            Vec3f w = normal;
            Vec3f u = (fabsf(w.x) > 0.1 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)).cross(w).normalized();
            Vec3f v = w.cross(u);
            Vec3f d = (u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2)).normalized();

            ray = Ray(hit_point, d);
            continue;
        }

        if (obj.reflection_type == ReflectionType::SPECULAR) { // Ideal SPECULAR reflection
            ray = Ray(hit_point, ray.d - surface_normal * 2 * surface_normal.dot(ray.d));
            continue;
        }

        Ray spawn_ray_reflect(hit_point,
                              ray.d - surface_normal * 2 *
                                      surface_normal.dot(ray.d)); // Ideal dielectric REFRACTION

        bool into = surface_normal.dot(normal) > 0; // Ray from outside going in?
        float nc = 1;
        float nt = 1.5;
        float nnt = into ? nc / nt : nt / nc;
        float ddn = ray.d.dot(normal);
        float cos2t = 1 - nnt * nnt * (1 - ddn * ddn);

        if (cos2t < 0) { // Total internal reflection
            ray = spawn_ray_reflect;
            continue;
        }

        Vec3f t_dir =
                (ray.d * nnt - surface_normal * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t)))).normalized();
        float a = nt - nc;
        float b = nt + nc;
        float R0 = a * a / (b * b);
        float c = 1 - (into ? -ddn : t_dir.dot(surface_normal));

        float Re = R0 + (1 - R0) * c * c * c * c * c;
        float Tr = 1 - Re;
        float probability_reflect = 0.25 + 0.5 * Re;

        float RP = Re / probability_reflect;
        float TP = Tr / (1 - probability_reflect);

        // refract or reflect
        if (sampler.generate() < probability_reflect) {
            // reflect
            ray = spawn_ray_reflect;
            throughput *= RP;
            continue;
        }

        // refract
        ray = Ray(hit_point, t_dir); // Ideal dielectric REFRACTION
        throughput *= TP;
        continue;
    }

    return radiance;
}

__global__
void render(Vec3f *frame_buffer, const int width, const int height, const int num_samples, const Sphere *spheres,
            const int num_spheres) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int flat_idx = (height - 1 - y) * width + x;

    Ray cam(Vec3f(50, 52, 295.6) * SCALER, Vec3f(0, -0.042612, -1).normalized()); // cam pos, dir
    Vec3f cx = Vec3f(width * 0.5135 / height, 0, 0);
    Vec3f cy = cx.cross(cam.d).normalized() * 0.5135;

    Sampler sampler(flat_idx);

    auto pixel_val = Vec3f(0.0, 0.0, 0.0);
    for (int s = 0; s < num_samples; s++) {
        float r1 = 2 * sampler.generate();
        float dx = r1 < 1 ? sqrtf(r1) - 1 : 1 - sqrtf(2 - r1);

        float r2 = 2 * sampler.generate();
        float dy = r2 < 1 ? sqrtf(r2) - 1 : 1 - sqrtf(2 - r2);

        Vec3f d = cx * ((dx + x) / width - 0.5f) + cy * ((dy + y - 20.f) / height - 0.5f) + cam.d;

        pixel_val += trace(Ray(cam.o + d * 140.f * SCALER, d.normalized()), spheres, num_spheres, sampler);
    }

    pixel_val = pixel_val * (1.0 / float(num_samples));

    frame_buffer[flat_idx] = Vec3f(pixel_val.x, pixel_val.y, pixel_val.z);
}

int main() {
    const float ratio = 1;
    const int width = 1024 * ratio;
    const int height = 768 * ratio;

    const int num_samples = 4096;

    Vec3f *frame_buffer;
    CUDA_CHECK_RETURN(cudaMallocManaged((void **) &frame_buffer, sizeof(Vec3f) * width * height));

    const int num_spheres = 9;
    Sphere *spheres;
    CUDA_CHECK_RETURN(cudaMallocManaged((void **) &spheres, sizeof(Sphere) * num_spheres));

    spheres[0].init(1e5, Vec3f(1e5 + 1, 40.8, 81.6), Vec3f(0, 0, 0), Vec3f(.75, .25, .25),
                    ReflectionType::DIFFUSE); // Left
    spheres[1].init(1e5, Vec3f(-1e5 + 99, 40.8, 81.6), Vec3f(0, 0, 0), Vec3f(.25, .25, .75),
                    ReflectionType::DIFFUSE);  // Right

    spheres[2].init(1e5, Vec3f(50, 40.8, 1e5), Vec3f(0, 0, 0), Vec3f(.75, .75, .75),
                    ReflectionType::DIFFUSE); // Back

    spheres[3].init(1e5, Vec3f(50, 40.8, -1e5 + 170), Vec3f(0, 0, 0), Vec3f(0, 0, 0),
                    ReflectionType::DIFFUSE); // Front
    spheres[4].init(1e5, Vec3f(50, 1e5, 81.6), Vec3f(0, 0, 0), Vec3f(.75, .75, .75),
                    ReflectionType::DIFFUSE); // Bottom
    spheres[5].init(1e5, Vec3f(50, -1e5 + 81.6, 81.6), Vec3f(0, 0, 0), Vec3f(.75, .75, .75),
                    ReflectionType::DIFFUSE); // Top
    spheres[6].init(16.5, Vec3f(27, 16.5, 47), Vec3f(0, 0, 0), Vec3f(1, 1, 1) * .999,
                    ReflectionType::SPECULAR); // Mirror
    spheres[7].init(16.5, Vec3f(73, 16.5, 78), Vec3f(0, 0, 0), Vec3f(1, 1, 1) * .999,
                    ReflectionType::REFRACTIVE); // Glass
    spheres[8].init(600, Vec3f(50, 681.6 - .27, 81.6), Vec3f(12, 12, 12), Vec3f(0, 0, 0),
                    ReflectionType::DIFFUSE); // Lite

    TicToc timer;
    timer.tic();

    constexpr int thread_width = 8;
    constexpr int thread_height = 8;

    dim3 threads(thread_width, thread_height);
    dim3 blocks(width / thread_width + 1, height / thread_height + 1);

    render<<<blocks, threads>>>(frame_buffer, width, height, num_samples, spheres, num_spheres);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    std::cout << "rendering (" << num_samples << " spp) took " << std::fixed << std::setprecision(3)
              << timer.toc() << " ms.\n"
              << std::flush;

    std::vector<unsigned char> png_pixels(width * height * 4);

    for (int i = 0; i < width * height; i++) {
        png_pixels[4 * i + 0] = toInt(frame_buffer[i].x);
        png_pixels[4 * i + 1] = toInt(frame_buffer[i].y);
        png_pixels[4 * i + 2] = toInt(frame_buffer[i].z);
        png_pixels[4 * i + 3] = 255;
    }

    std::string file_name = "cuda_pt_" + std::to_string(num_samples) + ".png";

    // Encode the image
    // if there's an error, display it
    if (unsigned error = lodepng::encode(file_name, png_pixels, width, height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }

    printf("image saved to `%s`\n", file_name.c_str());

    CUDA_CHECK_RETURN(cudaFree(frame_buffer));
    CUDA_CHECK_RETURN(cudaFree(spheres));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cudaDeviceReset();

    return 0;
}