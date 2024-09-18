/**
 * Generate camera rays from camera intrinsics / extrinsics
 * @date: 5.5.2024
 * @author: Qianyue He 
*/
#pragma once
#include "core/so3.cuh"
#include "core/ray.cuh"
#include "core/sampler.cuh"
#include <sstream>
#include <curand.h>
#include <tinyxml2.h>

// convert fov in degree to focal length
CPT_CPU_GPU float fov2focal(float fov, float img_size);
// convert focal length to degree angle 
CPT_CPU_GPU float focal2fov(float focal, float img_size);

CPT_CPU Vec3 parseVec3(const std::string& str);

class DeviceCamera {
public:
    SO3 R;              // camera rotation
    Vec3 t;        // camera translation (world frame) and orientation
    float inv_focal;    // focal length
    float _hw, _hh;     // pixel plane 
    Vec2 signs;
    bool use_orthogonal;
public:
    CPT_CPU_GPU DeviceCamera() {}

    CPT_CPU_GPU DeviceCamera(const Vec3& from, const Vec3& lookat, float fov, float w, float h, float hsign = 1, float vsign = 1);

    /**
     * Sampling ray with stratified sampling
    */
    CPT_GPU Ray generate_ray(int x, int y, Sampler& sampler) const {
        float x_pos = sampler.next1D() + float(x),
                y_pos = sampler.next1D() + float(y);
        float ndc_dir_x = (x_pos - _hw) * inv_focal * signs.x();
        float ndc_dir_y = (y_pos - _hh) * inv_focal * signs.y();
        Vec3 origin = t + use_orthogonal * (R.col(1) * ndc_dir_y + R.col(0) * ndc_dir_x);
        return Ray(origin, R.rotate(Vec3(use_orthogonal ? 0 : ndc_dir_x, use_orthogonal ? 0 : ndc_dir_y, 1.f)));
    }

    CPT_GPU Ray generate_ray(int x, int y, Vec2&& sample) const {
        float x_pos = sample.x() + float(x),
                y_pos = sample.y() + float(y);
        float ndc_dir_x = (x_pos - _hw) * inv_focal * signs.x();
        float ndc_dir_y = (y_pos - _hh) * inv_focal * signs.y();
        Vec3 origin = t + use_orthogonal * (R.col(1) * ndc_dir_y + R.col(0) * ndc_dir_x);
        return Ray(origin, R.rotate(Vec3(use_orthogonal ? 0 : ndc_dir_x, use_orthogonal ? 0 : ndc_dir_y, 1.f)));
    }

    CPT_CPU static DeviceCamera from_xml(const tinyxml2::XMLElement* sensorElement);

    CPT_CPU void move_forward(float step = 0.1) {
        t += step * R.rotate(Vec3(0, 0, 1));
    }

    CPT_CPU void move_backward(float step = 0.1) {
        t += step * R.rotate(Vec3(0, 0, -1));
    }

    CPT_CPU void move_left(float step = 0.1) {
        t += step * R.rotate(Vec3(-signs.x(), 0, 0));
    }

    CPT_CPU void move_right(float step = 0.1) {
        t += step * R.rotate(Vec3(signs.x(), 0, 0));
    }

    CPT_CPU void rotate(float yaw, float pitch);
};
