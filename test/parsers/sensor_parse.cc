#include <tinyxml2.h>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>

class Vec3 {
public:
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

// Helper function to parse a string of numbers into a Vec3
Vec3 parseVec3(const std::string& str) {
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

class Camera {
public:
    float fov;
    int sample_count;
    int max_bounce;
    Vec3 lookat_target;
    Vec3 lookat_origin;
    Vec3 lookat_up;
    int width;
    int height;

    Camera() : fov(0), sample_count(0), max_bounce(0), width(0), height(0) {}

    static Camera fromXML(const tinyxml2::XMLElement* sensorElement) {
        Camera camera;

        const tinyxml2::XMLElement* element = nullptr;

        // Read float and integer values
        element = sensorElement->FirstChildElement("float");
        while (element) {
            std::string name = element->Attribute("name");
            if (name == "fov") {
                element->QueryFloatAttribute("value", &camera.fov);
            }
            element = element->NextSiblingElement("float");
        }

        element = sensorElement->FirstChildElement("integer");
        while (element) {
            std::string name = element->Attribute("name");
            if (name == "sample_count") {
                element->QueryIntAttribute("value", &camera.sample_count);
            } else if (name == "max_bounce") {
                element->QueryIntAttribute("value", &camera.max_bounce);
            }
            element = element->NextSiblingElement("integer");
        }

        // Read transform values
        element = sensorElement->FirstChildElement("transform");
        if (element) {
            const tinyxml2::XMLElement* lookatElement = element->FirstChildElement("lookat");
            if (lookatElement) {
                camera.lookat_target = parseVec3(lookatElement->Attribute("target"));
                camera.lookat_origin = parseVec3(lookatElement->Attribute("origin"));
                camera.lookat_up = parseVec3(lookatElement->Attribute("up"));
            }
        }

        // Read film values
        element = sensorElement->FirstChildElement("film");
        if (element) {
            const tinyxml2::XMLElement* filmElement = element->FirstChildElement("integer");
            while (filmElement) {
                std::string name = filmElement->Attribute("name");
                if (name == "width") {
                    filmElement->QueryIntAttribute("value", &camera.width);
                } else if (name == "height") {
                    filmElement->QueryIntAttribute("value", &camera.height);
                }
                filmElement = filmElement->NextSiblingElement("integer");
            }
        }

        return camera;
    }
};

int main(int argc, char** argv) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(argv[1]) != tinyxml2::XML_SUCCESS) {
        std::cerr << "Failed to load file" << std::endl;
        return -1;
    }

    const tinyxml2::XMLElement* sceneElement = doc.FirstChildElement("scene");
    const tinyxml2::XMLElement* sensorElement = sceneElement->FirstChildElement("sensor");

    if (sensorElement) {
        Camera camera = Camera::fromXML(sensorElement);
        std::cout << "FOV: " << camera.fov << std::endl;
        std::cout << "Sample Count: " << camera.sample_count << std::endl;
        std::cout << "Max Bounce: " << camera.max_bounce << std::endl;
        std::cout << "LookAt Target: (" << camera.lookat_target.x << ", " << camera.lookat_target.y << ", " << camera.lookat_target.z << ")" << std::endl;
        std::cout << "LookAt Origin: (" << camera.lookat_origin.x << ", " << camera.lookat_origin.y << ", " << camera.lookat_origin.z << ")" << std::endl;
        std::cout << "LookAt Up: (" << camera.lookat_up.x << ", " << camera.lookat_up.y << ", " << camera.lookat_up.z << ")" << std::endl;
        std::cout << "Film Width: " << camera.width << std::endl;
        std::cout << "Film Height: " << camera.height << std::endl;
    }

    return 0;
}