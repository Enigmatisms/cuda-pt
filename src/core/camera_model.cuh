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
CPT_CPU_GPU float fov2focal(float fov, float img_size) {
    fov = fov / 180.f * M_PI;
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

    CPT_CPU static DeviceCamera from_xml(const tinyxml2::XMLElement* sensorElement) {
        float fov = 0;
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
        return DeviceCamera(lookat_origin, lookat_target, fov, width, height);
    }
};
