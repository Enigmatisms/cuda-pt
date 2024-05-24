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
#include "core/object.cuh"

using Vec3Arr = std::vector<Vec3>;
using Vec2Arr = std::vector<Vec2>;

CPT_CPU std::string getFolderPath(const char* filePath);

CPT_CPU Vec3 parseColor(const std::string& value);

CPT_CPU void parseBSDF(const tinyxml2::XMLElement* bsdf_elem, std::unordered_map<std::string, int>& bsdf_map, BSDF** bsdfs, int index);

CPT_CPU int getBSDFId(const std::unordered_map<std::string, int>& bsdfIdMap, const std::string& id);

void parseShape(
    const tinyxml2::XMLElement* shapeElement, 
    const std::unordered_map<std::string, int>& bsdf_map,
    std::vector<ObjInfo>& objects, std::array<Vec3Arr, 3>& verticesList, 
    std::array<Vec3Arr, 3>& normalsList, std::array<Vec2Arr, 3>& uvsList, 
    int prim_offset, std::string folder_prefix
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
            bsdf_id = getBSDFId(bsdf_map, id);
        } else if (type == "emitter") {

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
        ObjInfo object(bsdf_id, prim_offset, num_primitives, );
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