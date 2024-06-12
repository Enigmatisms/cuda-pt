/**
 * Scene parser (from xml)
 * @author: Qianyue He
 * @date:   2024.5.24
*/
#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <tinyxml2.h>
#include <unordered_map>
#include <cuda_runtime.h>
#include <tiny_obj_loader.h>
#include "core/config.cuh"
#include "core/aos.cuh"
#include "core/bsdf.cuh"
#include "core/shapes.cuh"
#include "core/object.cuh"
#include "core/emitter.cuh"
#include "core/camera_model.cuh"
#include "core/virtual_funcs.cuh"

using Vec4Arr = std::vector<Vec4>;
using Vec3Arr = std::vector<Vec3>;
using Vec2Arr = std::vector<Vec2>;

std::string getFolderPath(std::string filePath) {
    size_t pos = filePath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filePath.substr(0, pos + 1); // includes the last '/'
    }
    return ""; // include empty str if depth is 0
}

Vec4 parseColor(const std::string& value) {
    float r, g, b;
    if (value[0] == '#') {
        std::stringstream ss;
        ss << std::hex << value.substr(1);
        unsigned int color;
        ss >> color;
        r = float((color >> 16) & 0xFF) / 255.0f;
        g = float((color >> 8) & 0xFF) / 255.0f;
        b = float(color & 0xFF) / 255.0f;
    } else if (value.find(',') != std::string::npos || value.find(' ') != std::string::npos) {
        std::stringstream ss(value);
        std::vector<float> values;
        float component;
        while (ss >> component) {
            values.push_back(component);
            if (ss.peek() == ',' || ss.peek() == ' ') {
                ss.ignore();
            }
        }
        r = values[0];
        g = values[1];
        b = values[2];
    } else {
        std::stringstream ss(value);
        ss >> r;
        g = r;
        b = r;
    }
    return Vec4(r, g, b);
}

Vec3 parsePoint(const tinyxml2::XMLElement* element) {
    if (element == nullptr) {
        std::cerr << "Point not specified for point source.\n";
        throw std::runtime_error("Point element is null");
    }

    const char* name = element->Attribute("name");
    if (name == nullptr) {
        throw std::runtime_error("No 'name' attribute found");
    }

    float x = 0, y = 0, z = 0;
    tinyxml2::XMLError eResult = element->QueryFloatAttribute("x", &x);
    if (eResult != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("Error parsing 'x' attribute");
    }

    eResult = element->QueryFloatAttribute("y", &y);
    if (eResult != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("Error parsing 'y' attribute");
    }

    eResult = element->QueryFloatAttribute("z", &z);
    if (eResult != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("Error parsing 'z' attribute");
    }

    return Vec3(x, y, z);
}

void parseBSDF(const tinyxml2::XMLElement* bsdf_elem, std::unordered_map<std::string, int>& bsdf_map, BSDF** bsdfs, int index) {
    std::string type = bsdf_elem->Attribute("type");
    std::string id = bsdf_elem->Attribute("id");

    bsdf_map[id] = index;
    Vec4 k_d, k_s, k_g;
    int kd_tex_id = -1, ex_tex_id = -1;

    const tinyxml2::XMLElement* element = bsdf_elem->FirstChildElement("rgb");
    while (element) {
        std::string name = element->Attribute("name");
        std::string value = element->Attribute("value");
        Vec4 color = parseColor(value);
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

void parseEmitterNames(
    const tinyxml2::XMLElement* emitter_elem, 
    std::unordered_map<std::string, int>& emitter_map
) {
    int idx = 1;
    while (emitter_elem) {
        std::string id = emitter_elem->Attribute("id");
        emitter_map[id] = idx++;
        emitter_elem = emitter_elem->NextSiblingElement("emitter");
    }
}

void parseEmitter(
    const tinyxml2::XMLElement* emitter_elem, 
    std::unordered_map<std::string, int>& emitter_obj_map,      // key emitter name, value object_id
    std::vector<std::string> obj_ref_names,
    Emitter** emitters, int index
) {
    std::string type = emitter_elem->Attribute("type");
    std::string id = emitter_elem->Attribute("id");

    obj_ref_names.push_back(id);
    Vec4 emission(0, 0, 0), scaler(0, 0, 0);

    const tinyxml2::XMLElement* element = emitter_elem->FirstChildElement("rgb");
    while (element) {
        std::string name = element->Attribute("name");
        std::string value = element->Attribute("value");
        Vec4 color = parseColor(value);
        if (name == "emission") {
            emission = color;
        } else if (name == "scaler") {
            scaler = color;
        }
        element = element->NextSiblingElement("rgb");
    }

    if (type == "point") {
        element = emitter_elem->FirstChildElement("point");
        Vec3 pos(0, 0, 0);
        std::string name = element->Attribute("name");
        if (name == "center" || name == "pos")
            pos = parsePoint(element);
        create_point_source<<<1, 1>>>(emitters[index], emission * scaler, pos);
    } else if (type == "spot") {
        element = emitter_elem->FirstChildElement("point");
        while (element) {
            std::string name = element->Attribute("name");
            Vec3 pos(0, 0, 0), dir(0, 0, 1);
            if (name == "pos") {
                pos = parsePoint(element);
            } else if (name == "dir") {
                dir = parsePoint(element);
            }
            element = element->NextSiblingElement("point");
        }
        // create Spot source
    } else if (type == "area") {
        element = emitter_elem->FirstChildElement("string");
        std::string attr_name = element->Attribute("name");
        if (!element || attr_name != "bind_type") {
            std::cerr << "Bound primitive is not specified for area source '" << id << "', name: "<< element->Attribute("name") << std::endl;
            throw std::runtime_error("Bound primitive is not specified for area source");
        }
        bool spherical_bound = element->Attribute("value") == std::string("sphere");
        create_area_source<<<1, 1>>>(emitters[index], emission * scaler, emitter_obj_map[id], spherical_bound);
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

void parseSphereShape(
    const tinyxml2::XMLElement* shapeElement, 
    const std::unordered_map<std::string, int>& bsdf_map,
    const std::unordered_map<std::string, int>& emitter_map,
    std::unordered_map<std::string, int>& emitter_obj_map,
    std::vector<ObjInfo>& objects, std::array<Vec3Arr, 3>& verts_list, 
    std::array<Vec3Arr, 3>& norms_list, std::array<Vec2Arr, 3>& uvs_list, 
    int& prim_offset, std::string folder_prefix, int index
) {
    int bsdf_id = -1, emitter_id = 0;

    const tinyxml2::XMLElement* element = shapeElement->FirstChildElement("ref");
    
    while (element) {
        std::string type = element->Attribute("type");
        std::string id = element->Attribute("id");
        if (type == "material") {
            bsdf_id = get_map_id(bsdf_map, id);
        } else if (type == "emitter") {
            emitter_id = get_map_id(emitter_map, id);
            emitter_obj_map[id] = index;
        }
        element = element->NextSiblingElement("ref");
    }

    float radius = 0;
    Vec3 center(0, 0, 0);
    element = shapeElement->FirstChildElement("point");
    std::string name = element->Attribute("name");
    if (name == "center" || name == "pos")
        center = parsePoint(element);

    element = shapeElement->FirstChildElement("float");
    name = element->Attribute("name");
    if (name == "r" || name == "radius") {
        element->QueryFloatAttribute("value", &radius);
    }
    verts_list[0].emplace_back(std::move(center));
    verts_list[1].emplace_back(Vec3(radius, radius, radius));
    verts_list[2].emplace_back(Vec3(0, 0, 0));

    for (int i = 0; i < 3; i++) {
        norms_list[i].emplace_back(0, 1, 0);
        uvs_list[i].emplace_back(0, 0);
    }

    objects.emplace_back(bsdf_id, prim_offset, 1, emitter_id);
    objects.back().setup(verts_list, false);
    ++ prim_offset;
}

void parseObjShape(
    const tinyxml2::XMLElement* shapeElement, 
    const std::unordered_map<std::string, int>& bsdf_map,
    const std::unordered_map<std::string, int>& emitter_map,
    std::unordered_map<std::string, int>& emitter_obj_map,
    std::vector<ObjInfo>& objects, std::array<Vec3Arr, 3>& verts_list, 
    std::array<Vec3Arr, 3>& norms_list, std::array<Vec2Arr, 3>& uvs_list, 
    int& prim_offset, std::string folder_prefix, int index
) {
    std::string filename, name;
    int bsdf_id = -1, emitter_id = 0;

    const tinyxml2::XMLElement* element = shapeElement->FirstChildElement("string");
    while (element) {
        name = element->Attribute("name");
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
            bsdf_id = get_map_id(bsdf_map, id);
        } else if (type == "emitter") {
            emitter_id = get_map_id(emitter_map, id);
            emitter_obj_map[id] = index;
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

    int num_new_primitive = 0;
    for (const auto& shape : shapes) {
        int new_vert_num = shape.mesh.indices.size();
        if (new_vert_num % 3)
            std::cerr << "Warning: the number of primitives is not divisible by 3.\n";
        num_new_primitive += new_vert_num / 3;
    }
    for (int i = 0; i < 3; i++) {
        verts_list[i].reserve(verts_list.size() + num_new_primitive);
        norms_list[i].reserve(norms_list.size() + num_new_primitive);
        uvs_list[i].reserve(uvs_list.size() + num_new_primitive);
    }
    for (const auto& shape : shapes) {
        size_t num_primitives = shape.mesh.indices.size() / 3;
        ObjInfo object(bsdf_id, prim_offset, num_primitives, emitter_id);
        prim_offset += num_primitives;

        for (size_t i = 0; i < num_primitives; ++i) {
            int prim_base = 3 * i;
            bool has_normal = false;
            for (int j = 0; j < 3; ++j) {
                const tinyobj::index_t& idx = shape.mesh.indices[prim_base + j];
                int index = idx.vertex_index * 3;
                verts_list[j].emplace_back(attrib.vertices[index], attrib.vertices[index + 1], attrib.vertices[index + 2]);

                if (idx.normal_index >= 0) {
                    has_normal = true;
                    index = 3 * idx.normal_index;
                    norms_list[j].emplace_back(attrib.normals[index], attrib.normals[index + 1], attrib.normals[index + 2]);
                }
                if (idx.texcoord_index >= 0) {
                    index = 2 * idx.normal_index;
                    uvs_list[j].emplace_back(attrib.texcoords[index], attrib.texcoords[index + 1]);
                } else {
                    uvs_list[j].emplace_back(0, 0);
                }
            }
            if (!has_normal) {      // compute normals ourselves
                printf("Normal vector not found in '%s' primitive %lu, computing yet normal direction is not guaranteed.\n", name.c_str(), i);
                Vec3 diff = verts_list[1][i] - verts_list[0][i];
                Vec3 normal = diff.cross(verts_list[2][i] - verts_list[0][i]).normalized();
                for (int j = 0; j < 3; j++) {
                    norms_list[j].push_back(normal);
                }
            }
        }

        object.setup(verts_list);
        objects.push_back(object);
    }
}

class Scene {
public:
    BSDF** bsdfs;
    Emitter** emitters;
    std::vector<ObjInfo> objects;
    std::vector<Shape> shapes;

    std::array<Vec3Arr, 3> verts_list;
    std::array<Vec3Arr, 3> norms_list;
    std::array<Vec2Arr, 3> uvs_list;

    RenderingConfig config;

    DeviceCamera cam;
    int num_bsdfs;
    int num_prims;
    int num_emitters;
    int num_objects;
public:
    Scene(std::string path): num_bsdfs(0), num_emitters(0), num_objects(0), num_prims(0) {
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) {
            std::cerr << "Failed to load file" << std::endl;
        }

        auto folder_prefix = getFolderPath(path);
        const tinyxml2::XMLElement *scene_elem   = doc.FirstChildElement("scene"),
                                   *bsdf_elem    = scene_elem->FirstChildElement("brdf"),
                                   *shape_elem   = scene_elem->FirstChildElement("shape"),
                                   *emitter_elem = scene_elem->FirstChildElement("emitter"),
                                   *sensor_elem  = scene_elem->FirstChildElement("sensor"), *ptr = nullptr;

        std::unordered_map<std::string, int> bsdf_map, emitter_map, emitter_obj_map;
        std::vector<std::string> emitter_names;
        emitter_names.reserve(9);
        emitter_map.reserve(9);
        bsdf_map.reserve(32);

        // ------------------------- (1) parse all the BSDF -------------------------
        
        ptr = bsdf_elem;
        for (; ptr != nullptr; ++ num_bsdfs)
            ptr = ptr->NextSiblingElement("brdf");
        CUDA_CHECK_RETURN(cudaMalloc(&bsdfs, sizeof(BSDF*) * num_bsdfs));
        for (int i = 0; i < num_bsdfs; i++) {
            parseBSDF(bsdf_elem, bsdf_map, bsdfs, i);
            bsdf_elem = bsdf_elem->NextSiblingElement("brdf");
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());


        // ------------------------- (2) parse emitter names -------------------------
        parseEmitterNames(emitter_elem, emitter_map);

        // ------------------------- (3) parse all objects -------------------------
        ptr = shape_elem;
        for (; ptr != nullptr; ++ num_objects)
            ptr = ptr->NextSiblingElement("shape");
        objects.reserve(num_objects);

        std::vector<bool> sphere_objs(num_objects, false);

        for (int i = 0; i < 3; i++) {
            verts_list[i].reserve(32);
            norms_list[i].reserve(32);
            uvs_list[i].reserve(32);
        }

        int prim_offset = 0;
        for (int i = 0; i < num_objects; i++) {
            std::string type = shape_elem->Attribute("type");
            if (type == "obj")
                parseObjShape(shape_elem, bsdf_map, emitter_map, emitter_obj_map, objects, 
                            verts_list, norms_list, uvs_list, prim_offset, folder_prefix, i);
            else if (type == "sphere")
                parseSphereShape(shape_elem, bsdf_map, emitter_map, emitter_obj_map, objects, 
                            verts_list, norms_list, uvs_list, prim_offset, folder_prefix, i);
            sphere_objs[i] = type == "sphere";
            shape_elem = shape_elem->NextSiblingElement("shape");
        }
        num_prims = prim_offset;


        //  ------------------------- (4) parse all emitters --------------------------
        ptr = emitter_elem;
        for (; ptr != nullptr; ++ num_emitters)
            ptr = ptr->NextSiblingElement("emitter");
        CUDA_CHECK_RETURN(cudaMalloc(&emitters, sizeof(Emitter*) * (num_emitters + 1)));
        create_abstract_source<<<1, 1>>>(emitters[0]);
        for (int i = 1; i <= num_emitters; i++) {
            parseEmitter(emitter_elem, emitter_obj_map, emitter_names, emitters, i);
            emitter_elem = emitter_elem->NextSiblingElement("emitter");
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        // ------------------------- (5) parse camera & scene config -------------------------
        cam = DeviceCamera::from_xml(sensor_elem);
        config = RenderingConfig::from_xml(sensor_elem);

        // ------------------------- (6) initialize shapes -------------------------
        shapes.resize(num_prims);
        prim_offset = 0;
        for (int obj_id = 0; obj_id < num_objects; obj_id ++) {
            prim_offset += objects[obj_id].prim_num;
            for (int i = objects[obj_id].prim_offset; i < prim_offset; i++) {
                if (!sphere_objs[obj_id]) {
                    shapes[i] = TriangleShape(obj_id);
                } else {
                    shapes[i] = SphereShape(obj_id);
                }
            }
        }
    }

    ~Scene() {
        clear_vector();
        destroy_gpu_alloc<<<1, num_bsdfs>>>(bsdfs);
        destroy_gpu_alloc<<<1, num_emitters + 1>>>(emitters);

        CUDA_CHECK_RETURN(cudaFree(bsdfs));
        CUDA_CHECK_RETURN(cudaFree(emitters));
    }

    void export_soa(AoS3<Vec3>& verts, AoS3<Vec3>& norms, AoS3<Vec2>& uvs) const {
        verts.from_vectors(verts_list[0], verts_list[1], verts_list[2]);
        norms.from_vectors(norms_list[0], norms_list[1], norms_list[2]);
        uvs.from_vectors(uvs_list[0], uvs_list[1], uvs_list[2]);
    }

    void clear_vector() noexcept {
        for (int i = 0; i < 3; i++) {
            verts_list[i].clear();
            norms_list[i].clear();
            uvs_list[i].clear();

            verts_list[i].shrink_to_fit();
            norms_list[i].shrink_to_fit();
            uvs_list[i].shrink_to_fit();
        }
    }

    void print() const noexcept {
        std::cout << " Scene:\n";
        std::cout << "\tNumber of objects: \t" << num_objects << std::endl;
        std::cout << "\tNumber of primitives: \t" << num_prims << std::endl;
        std::cout << "\tNumber of emitters: \t" << num_emitters << std::endl;
        std::cout << "\tNumber of BSDFs: \t" << num_bsdfs << std::endl;
        std::cout << std::endl;
        std::cout << "\tConfig: width:\t\t" << config.width << std::endl;
        std::cout << "\tConfig: height:\t\t" << config.height << std::endl;
        std::cout << "\tConfig: max depth:\t" << config.max_depth << std::endl;
        std::cout << "\tConfig: SPP:\t\t" << config.spp << std::endl;
        std::cout << "\tConfig: Gamma corr:\t" << config.gamma_correction << std::endl;
        std::cout << std::endl;
    }
};