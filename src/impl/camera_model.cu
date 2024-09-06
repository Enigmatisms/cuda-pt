/**
 * @file camera_model.cu
 * 
 * @author Qianyue He
 * @brief implementation for some of the functions
 * @date 2024-09-06
 * @copyright Copyright (c) 2024
 */

#include "core/camera_model.cuh"

CPT_CPU_GPU float fov2focal(float fov, float img_size) {
    fov = fov / 180.f * M_Pi;
    return 0.5f * img_size / std::tan(.5f * fov);
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
    float w, float h, float hsign, float vsign
): t(from), inv_focal(1.f / fov2focal(fov, w)), _hw(w * 0.5f), _hh(h * 0.5f), signs(hsign, vsign) {
    dir = (lookat - from).normalized();
    R   = rotation_between(Vec3(0, 0, 1), dir);      
}

CPT_CPU DeviceCamera DeviceCamera::from_xml(const tinyxml2::XMLElement* sensorElement) {
    float fov = 0, hsign = 1, vsign = 1;
    int width = 512, height = 512;
    Vec3 lookat_target;
    Vec3 lookat_origin;
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
    return DeviceCamera(lookat_origin, lookat_target, fov, width, height, hsign, vsign);
}