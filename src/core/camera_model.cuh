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

CPT_CPU Vec3 parseVec3(const std::string& str);

class DeviceCamera {
private:
    SO3 R;              // camera rotation
    Vec3 t, dir;        // camera translation (world frame) and orientation
    float inv_focal;    // focal length
    float _hw, _hh;     // pixel plane 
    Vec2 signs;
public:
    CPT_CPU_GPU DeviceCamera() {}

    CPT_CPU_GPU DeviceCamera(const Vec3& from, const Vec3& lookat, float fov, float w, float h, float hsign = 1, float vsign = 1);

    /**
     * Sampling ray with stratified sampling
    */
    CPT_GPU Ray generate_ray(int x, int y, Sampler& sampler) const {
        float x_pos = sampler.next1D() + float(x),
                y_pos = sampler.next1D() + float(y);
        Vec3 ndc_dir((x_pos - _hw) * inv_focal * signs.x(), (y_pos - _hh) * inv_focal * signs.y(), 1.f);
        return Ray(t, R.rotate(ndc_dir.normalized()));
    }

    CPT_GPU Ray generate_ray(int x, int y, Vec2&& sample) const {
        float x_pos = sample.x() + float(x),
                y_pos = sample.y() + float(y);
        Vec3 ndc_dir((x_pos - _hw) * inv_focal * signs.x(), (y_pos - _hh) * inv_focal * signs.y(), 1.f);
        return Ray(t, R.rotate(ndc_dir.normalized()));
    }

    CPT_CPU static DeviceCamera from_xml(const tinyxml2::XMLElement* sensorElement);
};
