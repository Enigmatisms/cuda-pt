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
 * @brief implementation for some of the functions
 * @date 2024.09.06
 */
#include "core/camera_model.cuh"
#include <cmath>

CPT_CPU float fov2focal(float fov, float img_size) {
    fov = fov / 180.f * M_Pi;
    return 0.5f * img_size / std::tan(.5f * fov);
}

CPT_CPU float focal2fov(float inv_focal, float half_size) {
    return 360.f * M_1_Pi * atanf(half_size * inv_focal);
}

CPT_CPU Vec3 parseVec3(const std::string &str) {
    std::stringstream ss(str);
    std::vector<float> values;
    float value;
    while (ss >> value) {
        values.push_back(value);
        if (ss.peek() == ',' || ss.peek() == ' ') {
            ss.ignore();
        }
    }
    if (values.size() == 3) {
        return Vec3(values[0], values[1], values[2]);
    }
    return Vec3();
}

CPT_CPU DeviceCamera::DeviceCamera(const Vec3 &from, const Vec3 &lookat,
                                   float fov, float w, float h, float hsign = 1,
                                   Vec3 up = Vec3(0, 1, 0), float aperture,
                                   float focus_dist)
    : t(from), inv_focal(1.f / fov2focal(fov, w)), _hw(w * 0.5f), _hh(h * 0.5f),
      sign_x(hsign), aperture_radius(aperture), focal_distance(focus_dist) {
    Vec3 forward = (lookat - from).normalized_h();
    up.normalize_h();
    Vec3 right = up.cross(forward).normalized_h();
    R = SO3(right, up, forward, false);

    // compute actual focal dist if not given
    if (focus_dist <= 0) {
        focal_distance = (lookat - from).length();
    }
}

CPT_CPU void DeviceCamera::rotate(float yaw, float pitch) {
    auto quat_yaw = Quaternion::angleAxis_host(yaw, Vec3(0, sign_x, 0)),
         quat_pit = Quaternion::angleAxis_host(pitch, Vec3(1, 0, 0));
    SO3 rot = SO3::from_quat(quat_yaw * quat_pit);
    R = R * rot;
    Vec3 forward = R.col(2).normalized_h(), right = R.col(0);
    right.y() = 0;
    right *= 1.f / sqrtf(right.x() * right.x() + right.z() * right.z());
    Vec3 up = -right.cross(forward).normalized_h();
    R = SO3(right, up, forward, false);
}

CPT_CPU DeviceCamera
DeviceCamera::from_xml(const tinyxml2::XMLElement *sensorElement) {
    float fov = 0, hsign = 1;
    int width = 512, height = 512;
    Vec3 lookat_target;
    Vec3 lookat_origin;
    Vec3 lookat_up(0, 1, 0);
    const tinyxml2::XMLElement *element = nullptr;

    // Read float and integer values
    element = sensorElement->FirstChildElement("float");
    while (element) {
        std::string name = element->Attribute("name");
        if (name == "fov") {
            element->QueryFloatAttribute("value", &fov);
        }
        element = element->NextSiblingElement("float");
    }

    // Read transform values
    element = sensorElement->FirstChildElement("transform");
    if (element) {
        const tinyxml2::XMLElement *lookatElement =
            element->FirstChildElement("lookat");
        if (lookatElement) {
            lookat_target = parseVec3(lookatElement->Attribute("target"));
            lookat_origin = parseVec3(lookatElement->Attribute("origin"));
            auto element_ptr = lookatElement->Attribute("up");
            if (element_ptr == NULL) {
                std::cout
                    << "Up vector no specified. Using default up vector [0, 1, "
                       "0] for camera\n";
            } else {
                lookat_up = parseVec3(element_ptr);
            }
        } else {
            std::cerr << "XML scene file error:\n";
            throw std::runtime_error(
                "Camera element contains no 'lookat' transform\n");
        }
    }

    element = sensorElement->FirstChildElement("bool");
    while (element) {
        std::string name = element->Attribute("name");
        std::string value = element->Attribute("value");
        if (name == "hflip") {
            if (value == "true")
                hsign = -hsign;
        }
        element = element->NextSiblingElement("bool");
    }

    // Read film values
    element = sensorElement->FirstChildElement("film");
    if (element) {
        const tinyxml2::XMLElement *filmElement =
            element->FirstChildElement("integer");
        while (filmElement) {
            std::string name = filmElement->Attribute("name");
            if (name == "width") {
                filmElement->QueryIntAttribute("value", &width);
            } else if (name == "height") {
                filmElement->QueryIntAttribute("value", &height);
            }
            filmElement = filmElement->NextSiblingElement("integer");
        }
    }
    return DeviceCamera(lookat_origin, lookat_target, fov, width, height, hsign,
                        lookat_up);
}
