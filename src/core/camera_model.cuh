// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * @author Qianyue He
 * @brief Camera Model Definition
 * @date Unknown
 */

#pragma once
#include "core/ray.cuh"
#include "core/sampling.cuh"
#include "core/so3.cuh"

#include <curand.h>
#include <sstream>
#include <tinyxml2.h>

// convert fov in degree to focal length
CPT_CPU float fov2focal(float fov, float img_size);
// convert focal length to degree angle
CPT_CPU float focal2fov(float focal, float img_size);

CPT_CPU Vec3 parseVec3(const std::string &str);

class DeviceCamera {
  public:
    SO3 R;           // camera rotation (3 * 3)
    Vec3 t;          // camera translation (world frame) and orientation (3)
    float inv_focal; // focal length (1)
    float _hw, _hh;  // pixel plane (2)
    float sign_x;    // flipping sign for x (1)
    float aperture_radius = 0.0f; // aperture, 0 means no DoF
    float focal_distance = 1.0f;

  public:
    CPT_CPU_GPU DeviceCamera() {}

    CPT_CPU DeviceCamera(const Vec3 &from, const Vec3 &lookat, float fov,
                         float w, float h, float hsign = 1,
                         Vec3 up = Vec3(0, 1, 0), float aperture = 0,
                         float focus_dist = 1);

    CPT_GPU Ray generate_ray(int x, int y, Sampler &sampler) const {
        float x_pos = sampler.next1D() + static_cast<float>(x),
              y_pos = sampler.next1D() + static_cast<float>(y);

        float ndc_dir_x = (x_pos - _hw) * inv_focal * sign_x;
        float ndc_dir_y = (_hh - y_pos) * inv_focal;

        if (focal_distance == 0) {
            // orthogonal camera
            Vec3 origin = t + R.col(1) * ndc_dir_y + R.col(0) * ndc_dir_x;
            return Ray(std::move(origin), R.rotate(Vec3(0, 0, 1)).normalized());
        } else {
            // perspective camera
            Vec3 dir = R.rotate(Vec3(ndc_dir_x, ndc_dir_y, 1.0f)).normalized();
            if (aperture_radius > 0.0f) { // apply DoF
                Vec3 focus_point = t + dir * focal_distance;

                Vec2 lens_sample =
                    sample_uniform_disk(sampler.next2D()) * aperture_radius;
                Vec3 aperture_offset =
                    R.col(0) * lens_sample.x() + R.col(1) * lens_sample.y();

                Vec3 new_origin = t + aperture_offset;
                return Ray(std::move(new_origin),
                           (focus_point - new_origin).normalized());
            } else { // no DoF
                return Ray(t, dir);
            }
        }
    }

    CPT_CPU static DeviceCamera
    from_xml(const tinyxml2::XMLElement *sensorElement);

    CPT_GPU bool get_splat_pixel(const Vec3 &ray_d, int &px, int &py) const {
        // TODO(heqianyue): Currently, orthogonal camera and camera with
        // aperture do not support splatting
        Vec3 local_dir = -R.transposed_rotate(ray_d);
        bool success = false;
        if (local_dir.z() > 1e-5) {
            local_dir *= 1.f / local_dir.z(); // inverse NDC
            px = floorf(_hw + local_dir.x() / (inv_focal * sign_x));
            py = floorf(_hh + local_dir.y() / (-inv_focal));
            success = px >= 0 && px < _hw * 2 && py >= 0 && py < _hh * 2;
        }
        return success;
    }

    CPT_CPU void move_forward(float step = 0.1) { t += step * R.col(2); }

    CPT_CPU void move_backward(float step = 0.1) { t -= step * R.col(2); }

    CPT_CPU void move_left(float step = 0.1) {
        t += (-step * sign_x) * R.col(0);
    }

    CPT_CPU void move_right(float step = 0.1) {
        t += (step * sign_x) * R.col(0);
    }

    CPT_CPU void rotate(float yaw, float pitch);
};
