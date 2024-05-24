#include "core/parse_obj.cuh"
#include "core/virtual_funcs.cuh"

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

void parseBSDF(const tinyxml2::XMLElement* bsdf_elem, std::unordered_map<std::string, int>& bsdf_map, BSDF** bsdfs, int index) {
    std::string type = bsdf_elem->Attribute("type");
    std::string id = bsdf_elem->Attribute("id");

    Vec3 k_d, k_s, k_g;
    int kd_tex_id = -1, ex_tex_id = -1;

    const tinyxml2::XMLElement* element = bsdf_elem->FirstChildElement("rgb");
    while (element) {
        std::string name = element->Attribute("name");
        std::string value = element->Attribute("value");
        Vec3 color = parseColor(value);
        if (name == "k_d") {
            k_d = color;
        } else if (name == "k_s") {
            k_s = color;
        } else if (name == "k_g") {
            k_g = color;
        }
        element = element->NextSiblingElement("rgb");
    }

    element = bsdf_elem->FirstChildElement("integer");

    while (element) {
        std::string name = element->Attribute("name");
        std::string value = element->Attribute("value");
        int ref_id = -1;            // TODO: map texture with niteger index
        if (name == "kd_tex_id") {
            kd_tex_id = ref_id;
        } else if (name == "ex_tex_id") {
            ex_tex_id = ref_id;
        }
        element = element->NextSiblingElement("rgb");
    }

    if (type == "lambertian") {
        create_bsdf<LambertianBSDF><<<1, 1>>>(bsdfs + index, k_d, k_s, k_g, kd_tex_id, ex_tex_id);
    } else if (type == "specular") {
        create_bsdf<SpecularBSDF><<<1, 1>>>(bsdfs + index, k_d, k_s, k_g, kd_tex_id, ex_tex_id);
    } else if (type == "det-refraction") {
        create_bsdf<TranslucentBSDF><<<1, 1>>>(bsdfs + index, k_d, k_s, k_g, kd_tex_id, ex_tex_id);
    }
}

int get_map_id(const std::unordered_map<std::string, int>& map, const std::string& id) {
    auto it = map.find(id);
    if (it != map.end()) {
        return it->second;
    } else {
        std::cerr << "Map has no key: '" << id << "'\n";
        throw std::runtime_error("Map has no key: '" + id + "'");
    }
    return 0;
}

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