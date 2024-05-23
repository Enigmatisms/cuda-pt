#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <tinyxml2.h>
#include <unordered_map>
#include <cuda_runtime.h>
#include <tiny_obj_loader.h>
#include "core/soa.cuh"
#include "core/bsdf.cuh"

std::string getFolderPath(const char* filePath) {
    std::string path(filePath);
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(0, pos + 1); // includes the last '/'
    }
    return ""; // include empty str if depth is 0
}

Vec3 parseColor(const std::string& value) {
    unsigned int r, g, b;
    if (value[0] == '#') {
        std::stringstream ss;
        ss << std::hex << value.substr(1);
        unsigned int color;
        ss >> color;
        r = (color >> 16) & 0xFF;
        g = (color >> 8) & 0xFF;
        b = color & 0xFF;
    } else {
        std::stringstream ss(value);
        std::vector<float> values;
        float component;
        while (ss >> component) {
            values.push_back(component);
            if (ss.peek() == ',' || ss.peek() == ' ') {
                ss.ignore();
            }
        }
        if (values.size() == 3) {
            r = static_cast<unsigned int>(values[0] * 255);
            g = static_cast<unsigned int>(values[1] * 255);
            b = static_cast<unsigned int>(values[2] * 255);
        } else {
            return Vec3();
        }
    }
    return Vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}

void parseBSDF(const tinyxml2::XMLElement* brdfElement, std::unordered_map<std::string, int>& bsdfIdMap, std::vector<BSDF*>& bsdfs) {
    std::string type = brdfElement->Attribute("type");
    std::string id = brdfElement->Attribute("id");
    BSDF* bsdf = nullptr;

    if (type == "lambertian") {
        bsdf = new LambertianBSDF();
    } else if (type == "specular") {
        bsdf = new SpecularBSDF();
    }

    if (bsdf) {
        const tinyxml2::XMLElement* element = brdfElement->FirstChildElement("rgb");
        while (element) {
            std::string name = element->Attribute("name");
            std::string value = element->Attribute("value");
            Vec3 color = parseColor(value);
            if (name == "k_d") {
                bsdf->k_d = color;
            } else if (name == "k_s") {
                bsdf->k_s = color;
            } else if (name == "k_g") {
                bsdf->k_g = color;
            }
            element = element->NextSiblingElement("rgb");
        }

        bsdfs.push_back(bsdf);
        bsdfIdMap[id] = bsdfs.size() - 1;
    }
}

// AABB (placeholder)
class AABB {
    // AABB placeholder
};

class ObjInfo {
private:
    AABB _aabb;
public:
    int bsdf_id;            // index of the current object
    int prim_offset;        // offset to the start of the primitives
    int prim_num;           // number of primitives
    uint8_t emitter_id;     // index to the emitter, 0xff means not an emitter

    ObjInfo() : bsdf_id(-1), prim_offset(0), prim_num(0), emitter_id(0xff) {}
};

int getBSDFId(const std::unordered_map<std::string, int>& bsdfIdMap, const std::string& id) {
    auto it = bsdfIdMap.find(id);
    if (it != bsdfIdMap.end()) {
        return it->second;
    }
    return -1;
}

void parseShape(
    const tinyxml2::XMLElement* shapeElement, 
    const std::unordered_map<std::string, int>& bsdfIdMap,
    std::vector<ObjInfo>& objects, std::vector<SoA3<Vec3>>& verticesList, 
    std::vector<SoA3<Vec3>>& normalsList, std::vector<SoA3<Vec2>>& uvsList, 
    std::string folder_prefix
) {
    std::string filename;
    int bsdf_id = -1;

    const tinyxml2::XMLElement* element = shapeElement->FirstChildElement("string");
    while (element) {
        std::string name = element->Attribute("name");
        if (name == "filename") {
            filename = folder_prefix + element->Attribute("value");
        }
        element = element->NextSiblingElement("string");
    }

    element = shapeElement->FirstChildElement("ref");
    while (element) {
        std::string type = element->Attribute("type");
        std::string id = element->Attribute("id");
        if (type == "material") {
            bsdf_id = getBSDFId(bsdfIdMap, id);
        }
        element = element->NextSiblingElement("ref");
    }

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
    if (!ret) {
        std::cerr << "Failed to load/parse .obj file: " << filename << std::endl;
        return;
    }

    for (const auto& shape : shapes) {
        size_t num_primitives = shape.mesh.indices.size() / 3;
        SoA3<Vec3> vertices(num_primitives);
        SoA3<Vec3> normals(num_primitives);
        SoA3<Vec2> uvs(num_primitives);
        ObjInfo object;
        object.bsdf_id = bsdf_id;
        object.prim_offset = 0;  //  dummy setting
        object.prim_num = num_primitives;

        for (size_t i = 0; i < num_primitives; ++i) {
            for (int j = 0; j < 3; ++j) {
                const tinyobj::index_t& idx = shape.mesh.indices[3 * i + j];
                if (j == 0) {
                    vertices.x[i] = Vec3(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2]);
                    if (idx.normal_index >= 0) {
                        normals.x[i] = Vec3(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2]);
                    }
                    if (idx.texcoord_index >= 0) {
                        uvs.x[i] = Vec2(attrib.texcoords[2 * idx.texcoord_index + 0], attrib.texcoords[2 * idx.texcoord_index + 1]);
                    }
                } else if (j == 1) {
                    vertices.y[i] = Vec3(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2]);
                    if (idx.normal_index >= 0) {
                        normals.y[i] = Vec3(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2]);
                    }
                    if (idx.texcoord_index >= 0) {
                        uvs.y[i] = Vec2(attrib.texcoords[2 * idx.texcoord_index + 0], attrib.texcoords[2 * idx.texcoord_index + 1]);
                    }
                } else if (j == 2) {
                    vertices.z[i] = Vec3(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2]);
                    if (idx.normal_index >= 0) {
                        normals.z[i] = Vec3(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2]);
                    }
                    if (idx.texcoord_index >= 0) {
                        uvs.z[i] = Vec2(attrib.texcoords[2 * idx.texcoord_index + 0], attrib.texcoords[2 * idx.texcoord_index + 1]);
                    }
                }
            }
        }

        objects.push_back(object);
        verticesList.push_back(vertices);
        normalsList.push_back(normals);
        uvsList.push_back(uvs);
    }
}