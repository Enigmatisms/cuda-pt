/**
 * Generate camera rays from camera intrinsics / extrinsics
 * @date: 5.5.2024
 * @author: Qianyue He 
*/
#pragma once
#include "core/so3.cuh"
#include "core/ray.cuh"
#include "core/sampler.cuh"
#include <curand.h>

// convert fov in degree to focal length
CPT_CPU_GPU float fov2focal(float fov, float img_size) {
    fov = fov / 180.f * M_PI;
    return 0.5f * img_size / std::tan(.5f * fov);
}

class DeviceCamera {
private:
    SO3 R;              // camera rotation
    Vec3 t, dir;        // camera translation (world frame) and orientation
    float inv_focal;    // focal length
    float _hw, _hh;     // pixel plane 
public:
    CPT_CPU_GPU DeviceCamera() {}

    CPT_CPU_GPU DeviceCamera(const Vec3& from, const Vec3& lookat, float fov, float w, float h): 
        t(from), inv_focal(1.f / fov2focal(fov, w)), _hw(w * 0.5f), _hh(h * 0.5f) {
        dir = (lookat - from).normalized();
        R   = rotation_between(Vec3(0, 0, 1), dir);      
    }

    /**
     * Sampling ray with stratified sampling
    */
    CPT_GPU Ray generate_ray(int x, int y, Sampler& sampler) const {
        float x_pos = sampler.next1D() + float(x),
              y_pos = sampler.next1D() + float(y);
        Vec3 ndc_dir((x_pos - _hw) * inv_focal, (y_pos - _hh) * inv_focal, 1.f);
        return Ray(t, R.rotate(ndc_dir.normalized()));
    }
};
