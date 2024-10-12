/**
 * @file camera_model.cu
 * 
 * @author Qianyue He
 * @brief implementation for some of the functions
 * @date 2024-09-06
 * @copyright Copyright (c) 2024
 */
#include <cmath>
#include "core/camera_model.cuh"

CPT_CPU_GPU float fov2focal(float fov, float img_size) {
    fov = fov / 180.f * M_Pi;
    return 0.5f * img_size / std::tan(.5f * fov);
}

CPT_CPU_GPU float focal2fov(float inv_focal, float half_size) {
    return 360.f * M_1_Pi * atanf(half_size * inv_focal);
}

CPT_CPU static Vec3 parseVec3(const std::string& str) {
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

CPT_CPU_GPU DeviceCamera::DeviceCamera(
    const Vec3& from, const Vec3& lookat, float fov, 
    float w, float h, float hsign, float vsign, Vec3 up
): t(from), inv_focal(1.f / fov2focal(fov, w)), _hw(w * 0.5f), _hh(h * 0.5f), signs(hsign, vsign), use_orthogonal(false) {
    Vec3 forward = (lookat - from).normalized_h();
    up.normalize_h();
    Vec3 right = up.cross(forward).normalized_h();
    R = SO3(right, up, forward, false);
}

CPT_CPU void DeviceCamera::rotate(float yaw, float pitch) {
    auto quat_yaw = Quaternion::angleAxis(yaw, Vec3(0, signs.x(), 0)),
            quat_pit = Quaternion::angleAxis(pitch, Vec3(-signs.y(), 0, 0));
    SO3 rot = SO3::from_quat(quat_yaw * quat_pit);
    R = R * rot;
    Vec3 forward = R.col(2).normalized_h(),
         right   = R.col(0);
    right.y() = 0;
    right *= 1.f / sqrtf(right.x() * right.x() + right.z() * right.z());
    Vec3 up = -right.cross(forward).normalized_h();
    R = SO3(right, up, forward, false);
}

CPT_CPU DeviceCamera DeviceCamera::from_xml(const tinyxml2::XMLElement* sensorElement) {
    float fov = 0, hsign = 1, vsign = 1;
    int width = 512, height = 512;
    Vec3 lookat_target;
    Vec3 lookat_origin;
    Vec3 lookat_up(0, 1, 0);
    const tinyxml2::XMLElement* element = nullptr;

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
        const tinyxml2::XMLElement* lookatElement = element->FirstChildElement("lookat");
        if (lookatElement) {
            lookat_target = parseVec3(lookatElement->Attribute("target"));
            lookat_origin = parseVec3(lookatElement->Attribute("origin"));
            auto element_ptr = lookatElement->Attribute("up");
            if (element_ptr == NULL) {
                std::cout << "Up vector no specified. Using default up vector [0, 1, 0] for camera\n";
            } else {
                lookat_up = parseVec3(element_ptr);
            }
        } else {
            std::cerr << "XML scene file error:\n";
            throw std::runtime_error("Camera element contains no 'lookat' transform\n");
        }
    }

    element = sensorElement->FirstChildElement("bool");
    while (element) {
        std::string name = element->Attribute("name");
        std::string value = element->Attribute("value");
        if(name == "vflip") {
            if (value == "true") vsign = -vsign;
        } else if (name == "hflip") {
            if (value == "true") hsign = -hsign;
        }
        element = element->NextSiblingElement("bool");
    }

    // Read film values
    element = sensorElement->FirstChildElement("film");
    if (element) {
        const tinyxml2::XMLElement* filmElement = element->FirstChildElement("integer");
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
    return DeviceCamera(lookat_origin, lookat_target, fov, width, height, hsign, vsign, lookat_up);
}