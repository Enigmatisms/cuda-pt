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
 * @author: Qianyue He
 * @brief Scene parser (from xml)
 * This is the implementation file
 * @date:   2024.9.6
 */
#include "core/scene.cuh"
#include "volume/phase_registry.cuh"
#include <numeric>

static constexpr int MAX_PRIMITIVE_NUM = 64000000;
static constexpr int MAX_ALLOWED_BSDF = 48;
static constexpr const char *SCENE_VERSION = "1.2";
using TypedVec4 = std::pair<Vec4, PhaseFuncType>;

const std::unordered_map<std::string, MetalType> conductor_mapping = {
    {"Au", MetalType::Au},     {"Cr", MetalType::Cr},   {"Cu", MetalType::Cu},
    {"Ag", MetalType::Ag},     {"Al", MetalType::Al},   {"W", MetalType::W},
    {"TiO2", MetalType::TiO2}, {"Ni", MetalType::Ni},   {"MgO", MetalType::MgO},
    {"Na", MetalType::Na},     {"SiC", MetalType::SiC}, {"V", MetalType::V},
    {"CuO", MetalType::CuO},   {"Hg", MetalType::Hg},   {"Ir", MetalType::Ir},
};

const std::unordered_map<std::string, DispersionType> dielectric_mapping = {
    {"Diamond", DispersionType::Diamond},
    {"DiamondHigh", DispersionType::DiamondHigh},
    {"Silica", DispersionType::Silica},
    {"Glass_BK7", DispersionType::Glass_BK7},
    {"Glass_BaF10", DispersionType::Glass_BaF10},
    {"Glass_SF10", DispersionType::Glass_SF10},
    {"Sapphire", DispersionType::Sapphire},
    {"Water", DispersionType::Water}};

static std::string get_folder_path(std::string filePath) {
    size_t pos = filePath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filePath.substr(0, pos + 1); // includes the last '/'
    }
    return ""; // include empty str if depth is 0
}

Vec4 parseColor(std::string value) {
    float r, g, b;
    if (value[0] == '#') {
        std::stringstream ss;
        ss << std::hex << value.substr(1);
        unsigned int color;
        ss >> color;
        r = float((color >> 16) & 0xFF) / 255.0f;
        g = float((color >> 8) & 0xFF) / 255.0f;
        b = float(color & 0xFF) / 255.0f;
    } else if (value.find(',') != std::string::npos ||
               value.find(' ') != std::string::npos) {
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

Vec3 parsePoint(const tinyxml2::XMLElement *element) {
    if (element == nullptr) {
        std::cerr << "Point not specified for point source.\n";
        throw std::runtime_error("Point element is null");
    }
    std::string err_what = "Point parsing failed.\n";
    float x = 0, y = 0, z = 0;
    if (element->QueryFloatAttribute("x", &x) != tinyxml2::XML_SUCCESS) {
        std::cerr
            << "Warning: point element 'x' attribute not set. Using default "
               "value 0.\n";
    }
    if (element->QueryFloatAttribute("y", &y) != tinyxml2::XML_SUCCESS) {
        std::cerr
            << "Warning: point element 'y' attribute not set. Using default "
               "value 0.\n";
    }
    if (element->QueryFloatAttribute("z", &z) != tinyxml2::XML_SUCCESS) {
        std::cerr
            << "Warning: point element 'z' attribute not set. Using default "
               "value 0.\n";
    }
    return Vec3(x, y, z);
}

template <typename Ty>
Ty extract_from(const tinyxml2::XMLElement *elem,
                std::string value_name = "value", std::string error_what = "") {
    tinyxml2::XMLError e_ret = tinyxml2::XML_SUCCESS;
    Ty result;
    if constexpr (std::is_same_v<std::decay_t<Ty>, bool>) {
        e_ret = elem->QueryBoolAttribute(value_name.c_str(), &result);
    } else if constexpr (std::is_same_v<std::decay_t<Ty>, float>) {
        e_ret = elem->QueryFloatAttribute(value_name.c_str(), &result);
    } else if constexpr (std::is_same_v<std::decay_t<Ty>, int>) {
        e_ret = elem->QueryIntAttribute(value_name.c_str(), &result);
    } else if constexpr (std::is_same_v<std::decay_t<Ty>, std::string>) {
        result = elem->Attribute(value_name.c_str());
    } else if constexpr (std::is_same_v<std::decay_t<Ty>, Vec4>) {
        result = parseColor(elem->Attribute(value_name.c_str()));
    } else if constexpr (std::is_same_v<std::decay_t<Ty>, Vec3>) {
        result = parsePoint(elem);
    } else {
        std::cerr << "Unsupported type '" << typeid(Ty).name()
                  << "' for extraction.\n";
        std::cerr << error_what;
        throw std::runtime_error("`extract_from` unsupported type.");
    }

    if (e_ret != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("`extract_from` parsing failed");
    }
    return result;
}

template <typename Ty>
bool parse_attribute(const tinyxml2::XMLElement *elem, Ty &inout_value,
                     std::initializer_list<std::string> &&names,
                     std::string error_what = "",
                     std::string value_name = "value",
                     std::string name_name = "name") {
    if (elem == nullptr) {
        return false;
    }
    bool name_check = false;
    const std::string elem_name = elem->Attribute(name_name.c_str());
    for (auto name : names) {
        if (name == elem_name) {
            name_check = true;
            break;
        }
    }
    if (!name_check)
        return false;
    inout_value = extract_from<Ty>(elem, value_name, error_what);
    return true;
}

void parseBSDF(const tinyxml2::XMLElement *bsdf_elem,
               const std::unordered_map<std::string, TextureInfo> &tex_map,
               std::unordered_map<std::string, int> &bsdf_map,
               std::vector<BSDFInfo> &bsdf_infos,
               std::vector<Texture<float4>> &host_4d,
               std::vector<Texture<float2>> &host_2d, Textures &textures,
               BSDF **bsdfs, int index) {
    std::string type = bsdf_elem->Attribute("type");
    std::string id = bsdf_elem->Attribute("id");

    bsdf_map[id] = index;
    Vec4 k_d, k_s, k_g;

    const tinyxml2::XMLElement *element = bsdf_elem->FirstChildElement("rgb");
    while (element) {
        std::string err_what = "[BSDF] Vec4 parsing failed.\n";
        parse_attribute<Vec4>(element, k_d, {"k_d"}, err_what);
        parse_attribute<Vec4>(element, k_s, {"k_s"}, err_what);
        parse_attribute<Vec4>(element, k_g, {"k_g", "sigma_a"}, err_what);
        element = element->NextSiblingElement("rgb");
    }

    // reference to the texture
    element = bsdf_elem->FirstChildElement("ref");
    if (element) {
        std::string name = element->Attribute("type");
        if (!name.empty() && name == "texture") {
            std::string value = element->Attribute("id");
            auto it = tex_map.find(value);
            if (it == tex_map.end()) {
                std::cerr << "Texture named '" << value << "' not found.\n";
                throw std::runtime_error("Referenced Texture not found.");
            } else {
                if (!it->second.diff_path.empty()) {
                    Texture<float4> tex(it->second.diff_path,
                                        TextureType::DIFFUSE_TEX);
                    textures.enqueue(tex, index);
                    host_4d.emplace_back(std::move(tex));
                }
                if (!it->second.spec_path.empty()) {
                    Texture<float4> tex(it->second.spec_path,
                                        TextureType::SPECULAR_TEX);
                    textures.enqueue(tex, index);
                    host_4d.emplace_back(std::move(tex));
                }
                if (!it->second.glos_path.empty()) {
                    Texture<float4> tex(it->second.glos_path,
                                        TextureType::GLOSSY_TEX);
                    textures.enqueue(tex, index);
                    host_4d.emplace_back(std::move(tex));
                }
                if (!it->second.rough_path1.empty()) {
                    Texture<float2> tex(
                        it->second.rough_path1, TextureType::ROUGHNESS_TEX,
                        it->second.rough_path2, it->second.is_rough_ior);
                    textures.enqueue(tex, index);
                    host_2d.emplace_back(std::move(tex));
                }
                if (!it->second.normal_path.empty()) {
                    Texture<float4> tex(it->second.normal_path,
                                        TextureType::NORMAL_TEX, "", false,
                                        true);
                    textures.enqueue(tex, index);
                    host_4d.emplace_back(std::move(tex));
                }
            }
        }
    }

    BSDFInfo info(id);
    info.bsdf = BSDFInfo::BSDFParams(k_d, k_s, k_g);
    if (type == "lambertian") {
        create_bsdf<LambertianBSDF><<<1, 1>>>(
            bsdfs + index, k_d, k_s, k_g,
            ScatterStateFlag::BSDF_DIFFUSE | ScatterStateFlag::BSDF_REFLECT);
    } else if (type == "specular") {
        create_bsdf<SpecularBSDF><<<1, 1>>>(bsdfs + index, k_d, k_s, k_g,
                                            ScatterStateFlag::BSDF_SPECULAR |
                                                ScatterStateFlag::BSDF_REFLECT);
        info.type = BSDFType::Specular;
    } else if (type == "det-refraction") {
        create_bsdf<TranslucentBSDF><<<1, 1>>>(
            bsdfs + index, k_d, k_s, k_g,
            ScatterStateFlag::BSDF_SPECULAR | ScatterStateFlag::BSDF_TRANSMIT);
        info.type = BSDFType::Translucent;
    } else if (type == "conductor-ggx") {
        float roughness_x = 0.1f, roughness_y = 0.1f;
        MetalType mtype = MetalType::Cu;
        info.type = BSDFType::GGXConductor;
        element = bsdf_elem->FirstChildElement("string");
        if (element) {
            std::string name = element->Attribute("name");
            if (name == "type" || name == "metal" || name == "conductor") {
                std::string metal_type = element->Attribute("value");
                auto it = conductor_mapping.find(metal_type);
                if (it == conductor_mapping.end()) {
                    std::cout << "BSDF[" << id << "]"
                              << ": Only << " << int(NumMetalType)
                              << " types of metals are supported: ";
                    for (const auto [k, v] : conductor_mapping)
                        std::cout << k << ", ";
                    std::cout << std::endl;
                    std::cout << "Current type '" << metal_type
                              << "' is not supported. Setting to 'Cu'\n";
                } else {
                    mtype = it->second;
                }
            }
        }
        element = bsdf_elem->FirstChildElement("float");
        std::string error_what = "[Conductor-GGX] Roughness parsing failed.\n";
        while (element) {
            if (parse_attribute<float>(element, roughness_x,
                                       {"roughness_x", "rough_x"},
                                       error_what)) {
                roughness_x = std::clamp(roughness_x, 0.001f, 1.f);
            } else if (parse_attribute<float>(element, roughness_y,
                                              {"roughness_y", "rough_y"},
                                              error_what)) {
                roughness_y = std::clamp(roughness_y, 0.001f, 1.f);
            }
            element = element->NextSiblingElement("float");
        }
        info.bsdf.store_ggx_params(mtype, k_g, roughness_x, roughness_y);
        create_metal_bsdf<<<1, 1>>>(bsdfs + index, METAL_ETA_TS[mtype],
                                    METAL_KS[mtype], k_g, roughness_x,
                                    roughness_y);
    } else if (type == "plastic" || type == "plastic-forward") {
        k_g = Vec4(0, 1);
        element = bsdf_elem->FirstChildElement("float");
        float trans_scaler = 1.f, thickness = 0.f, ior = 1.33f;
        std::string error_what = "[Plastic] BSDF Param parsing failed.\n";
        while (element) {
            parse_attribute<float>(element, trans_scaler, {"trans_scaler"},
                                   error_what);
            parse_attribute<float>(element, thickness, {"thickness"},
                                   error_what);
            parse_attribute<float>(element, ior, {"ior"}, error_what);
            element = element->NextSiblingElement("float");
        }
        bool penetrable = false;
        element = bsdf_elem->FirstChildElement("bool");
        parse_attribute<bool>(element, penetrable, {"penetrable"}, error_what);

        if (type == "plastic") {
            info.type = BSDFType::Plastic;
            create_plastic_bsdf<PlasticBSDF><<<1, 1>>>(bsdfs + index, k_d, k_s,
                                                       k_g, ior, trans_scaler,
                                                       thickness, penetrable);
        } else {
            info.type = BSDFType::PlasticForward;
            create_plastic_bsdf<PlasticForwardBSDF>
                <<<1, 1>>>(bsdfs + index, k_d, k_s, k_g, ior, trans_scaler,
                           thickness, penetrable);
        }
        info.bsdf.store_plastic_params(ior, trans_scaler, thickness);
    } else if (type == "dispersion") {
        element = bsdf_elem->FirstChildElement("string");
        DispersionType dtype = DispersionType::Diamond;
        if (element) {
            std::string name = element->Attribute("name");
            if (name == "type" || name == "dielectric") {
                std::string dielec_type = element->Attribute("value");
                auto it = dielectric_mapping.find(dielec_type);
                if (it == dielectric_mapping.end()) {
                    std::cout << "BSDF[" << id << "]"
                              << ": Only 8 types of metals are supported: ";
                    for (const auto &[k, v] : dielectric_mapping)
                        std::cout << k << ", ";
                    std::cout << std::endl;
                    std::cout << "Current type '" << dielec_type
                              << "' is not supported. Setting to 'Diamond'\n";
                } else {
                    dtype = it->second;
                }
            }
        }
        Vec2 dis_params = DISPERSION_PARAMS[dtype];
        info.type = BSDFType::Dispersion;
        info.bsdf.store_dispersion_params(dtype, k_s);
        create_dispersion_bsdf<<<1, 1>>>(bsdfs + index, k_s, dis_params.x(),
                                         dis_params.y());
    } else if (type == "forward") {
        create_forward_bsdf<<<1, 1>>>(bsdfs + index);
        info.type = BSDFType::Forward;
    }
    bsdf_infos.emplace_back(std::move(info));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void parseEmitterNames(const tinyxml2::XMLElement *emitter_elem,
                       std::unordered_map<std::string, int> &emitter_map) {
    int idx = 1;
    while (emitter_elem) {
        std::string id = emitter_elem->Attribute("id");
        emitter_map[id] = idx++;
        emitter_elem = emitter_elem->NextSiblingElement("emitter");
    }
}

void parseEmitter(const tinyxml2::XMLElement *emitter_elem,
                  std::unordered_map<std::string, int>
                      &emitter_obj_map, // key emitter name, value object_id,
                  const std::unordered_map<std::string, TextureInfo> &tex_map,
                  std::vector<std::pair<std::string, Vec4>> &e_props,
                  std::vector<std::string> obj_ref_names,
                  std::vector<Texture<float4>> &host_texs, Emitter **emitters,
                  int &envmap_id, int index) {
    std::string type = emitter_elem->Attribute("type");
    std::string id = emitter_elem->Attribute("id");

    obj_ref_names.push_back(id);
    Vec4 emission(0, 0, 0), scaler(1, 1, 1);

    const tinyxml2::XMLElement *element =
        emitter_elem->FirstChildElement("rgb");
    while (element) {
        parse_attribute<Vec4>(element, emission, {"emission"});
        parse_attribute<Vec4>(element, scaler, {"scaler"});
        element = element->NextSiblingElement("rgb");
    }
    scaler.w() = scaler.x();
    e_props.emplace_back(id, Vec4(emission.xyz(), scaler.x()));

    cudaTextureObject_t tex_obj = 0;
    element = emitter_elem->FirstChildElement("ref");
    if (element) {
        std::string name = element->Attribute("type");
        if (!name.empty() && name == "texture") {
            std::string value = element->Attribute("id");
            auto it = tex_map.find(value);
            if (it == tex_map.end()) {
                std::cerr << "Texture named '" << value << "' not found.\n";
                throw std::runtime_error("Referenced Texture not found.");
            } else {
                if (!it->second.diff_path.empty()) {
                    Texture<float4> tex(it->second.diff_path,
                                        TextureType::DIFFUSE_TEX);
                    tex_obj = tex.object();
                    host_texs.emplace_back(std::move(tex));
                } else {
                    std::cerr << "The texture for HDRI should be set for it's "
                                 "'emission' "
                                 "element, but none is found.\n";
                    throw std::runtime_error("Referenced Texture not found.");
                }
            }
        }
    }

    if (type == "point") {
        element = emitter_elem->FirstChildElement("point");
        Vec3 pos(0, 0, 0);
        parse_attribute(element, pos, {"center", "pos"});
        create_point_source<<<1, 1>>>(emitters[index], emission * scaler, pos);
    } else if (type == "area-spot") {
        element = emitter_elem->FirstChildElement("float");
        float cos_val = 0.99;
        if (parse_attribute(element, cos_val, {"half-angle", "angle"})) {
            cos_val = cosf(cos_val * DEG2RAD);
        }
        element = emitter_elem->FirstChildElement("string");
        std::string attr_name = element->Attribute("name");
        if (!element || attr_name != "bind_type") {
            std::cerr
                << "Bound primitive is not specified for area spot source '"
                << id << "', name: " << element->Attribute("name") << std::endl;
            throw std::runtime_error(
                "Bound primitive is not specified for area spot source");
        }
        bool spherical_bound =
            element->Attribute("value") == std::string("sphere");
        create_area_spot_source<<<1, 1>>>(emitters[index], emission * scaler,
                                          cos_val, emitter_obj_map[id],
                                          spherical_bound, tex_obj);
    } else if (type == "area") {
        element = emitter_elem->FirstChildElement("string");
        std::string attr_name = element->Attribute("name");
        if (!element || attr_name != "bind_type") {
            std::cerr << "Bound primitive is not specified for area source '"
                      << id << "', name: " << element->Attribute("name")
                      << std::endl;
            throw std::runtime_error(
                "Bound primitive is not specified for area source");
        }
        bool spherical_bound =
            element->Attribute("value") == std::string("sphere");
        create_area_source<<<1, 1>>>(emitters[index], emission * scaler,
                                     emitter_obj_map[id], spherical_bound,
                                     tex_obj);
    } else if (type == "envmap") {
        envmap_id = index;
        element = emitter_elem->FirstChildElement("float");
        float scaler = 1.f, azimuth = 0.f, zenith = 0.f;
        while (element) {
            parse_attribute(element, scaler, {"scaler"});
            parse_attribute(element, azimuth, {"azimuth"});
            parse_attribute(element, zenith, {"zenith"});
            element = element->NextSiblingElement("float");
        }
        e_props.back().second = Vec4(-1, scaler, azimuth, zenith);
        element = emitter_elem->FirstChildElement("ref");
        if (tex_obj != 0) {
            create_envmap_source<<<1, 1>>>(emitters[index], tex_obj, scaler,
                                           azimuth * DEG2RAD, zenith * DEG2RAD);
        } else {
            std::cerr << "Error: The texture for EnvMap is empty.\n";
            throw std::runtime_error("Referenced Texture not available.");
        }
    }
}

int get_map_id(const std::unordered_map<std::string, int> &map,
               const std::string &id) {
    auto it = map.find(id);
    if (it != map.end()) {
        return it->second;
    } else {
        std::cerr << "Map has no key: '" << id << "'\n";
        throw std::runtime_error("Map has no key: '" + id + "'");
    }
    return 0;
}

void parseSphereShape(const tinyxml2::XMLElement *shapeElement,
                      const std::unordered_map<std::string, int> &bsdf_map,
                      const std::unordered_map<std::string, int> &emitter_map,
                      std::vector<BSDFInfo> &bsdf_infos,
                      std::unordered_map<std::string, int> &emitter_obj_map,
                      std::vector<ObjInfo> &objects,
                      std::array<Vec3Arr, 3> &verts_list,
                      std::array<Vec3Arr, 3> &norms_list,
                      std::array<Vec2Arr, 3> &uvs_list, int &prim_offset,
                      std::string folder_prefix, int index) {
    int bsdf_id = -1, emitter_id = 0;

    const tinyxml2::XMLElement *element =
        shapeElement->FirstChildElement("ref");

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
    if (bsdf_id == -1) {
        std::cerr << "The current object does not have an attached BSDF.\n";
        throw std::runtime_error("Object with no BSDF");
    }
    bsdf_infos[bsdf_id].in_use = true;

    float radius = 0;
    Vec3 center(0, 0, 0);
    element = shapeElement->FirstChildElement("point");
    parse_attribute(element, center, {"center", "pos"});

    element = shapeElement->FirstChildElement("float");
    parse_attribute(element, radius, {"radius", "r"});

    verts_list[0].emplace_back(std::move(center));
    verts_list[1].emplace_back(radius, radius, radius);
    verts_list[2].emplace_back(0, 0, 0);

    for (int i = 0; i < 3; i++) {
        norms_list[i].emplace_back(0, 1, 0);
        uvs_list[i].emplace_back(0, 0);
    }

    objects.emplace_back(bsdf_id, prim_offset, 1, emitter_id);
    objects.back().setup(verts_list, false);
    ++prim_offset;
}

void parseObjShape(const tinyxml2::XMLElement *shapeElement,
                   const std::unordered_map<std::string, int> &bsdf_map,
                   const std::unordered_map<std::string, int> &emitter_map,
                   std::vector<BSDFInfo> &bsdf_infos,
                   std::unordered_map<std::string, int> &emitter_obj_map,
                   std::vector<ObjInfo> &objects,
                   std::array<Vec3Arr, 3> &verts_list,
                   std::array<Vec3Arr, 3> &norms_list,
                   std::array<Vec2Arr, 3> &uvs_list, int &prim_offset,
                   std::string folder_prefix, int index) {
    std::string filename, name;
    int bsdf_id = -1, emitter_id = 0;

    const tinyxml2::XMLElement *element =
        shapeElement->FirstChildElement("string");
    while (element) {
        if (parse_attribute(element, filename, {"filename"})) {
            filename = folder_prefix + filename;
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
    if (bsdf_id == -1) {
        std::cerr << "The current object does not have an attached BSDF.\n";
        throw std::runtime_error("Object with no BSDF");
    }
    bsdf_infos[bsdf_id].in_use = true;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                filename.c_str());
    if (!ret) {
        std::cerr << "Failed to load/parse .obj file: " << filename
                  << std::endl;
        return;
    }

    int num_new_primitive = 0;
    for (const auto &shape : shapes) {
        int new_vert_num = shape.mesh.indices.size();
        if (new_vert_num % 3)
            std::cerr
                << "Warning: the number of primitives is not divisible by 3.\n";
        num_new_primitive += new_vert_num / 3;
    }
    for (int i = 0; i < 3; i++) {
        verts_list[i].reserve(verts_list.size() + num_new_primitive);
        norms_list[i].reserve(norms_list.size() + num_new_primitive);
        uvs_list[i].reserve(uvs_list.size() + num_new_primitive);
    }
    ObjInfo object(bsdf_id, prim_offset, num_new_primitive, emitter_id);
    for (const auto &shape : shapes) {
        size_t num_primitives = shape.mesh.indices.size() / 3;
        prim_offset += num_primitives;

        for (size_t i = 0; i < num_primitives; ++i) {
            int prim_base = 3 * i;
            bool has_normal = false;
            for (int j = 0; j < 3; ++j) {
                const tinyobj::index_t &idx = shape.mesh.indices[prim_base + j];
                int index = idx.vertex_index * 3;
                verts_list[j].emplace_back(attrib.vertices[index],
                                           attrib.vertices[index + 1],
                                           attrib.vertices[index + 2]);

                if (idx.normal_index >= 0) {
                    has_normal = true;
                    index = 3 * idx.normal_index;
                    norms_list[j].emplace_back(attrib.normals[index],
                                               attrib.normals[index + 1],
                                               attrib.normals[index + 2]);
                }
                if (idx.texcoord_index >= 0) {
                    index = 2 * idx.texcoord_index;
                    uvs_list[j].emplace_back(attrib.texcoords[index],
                                             attrib.texcoords[index + 1]);
                } else {
                    uvs_list[j].emplace_back(0, 0);
                }
            }
            if (!has_normal) { // compute normals ourselves
                printf("Normal vector not found in '%s' primitive %lu, "
                       "computing yet "
                       "normal direction is not guaranteed.\n",
                       name.c_str(), i);
                Vec3 diff = verts_list[1][i] - verts_list[0][i];
                Vec3 normal = diff.cross(verts_list[2][i] - verts_list[0][i])
                                  .normalized_h();
                for (int j = 0; j < 3; j++) {
                    norms_list[j].push_back(normal);
                }
            }
        }
    }
    object.setup(verts_list);
    objects.push_back(object);
}

void parseTexture(const tinyxml2::XMLElement *tex_elem,
                  std::unordered_map<std::string, TextureInfo> &texs,
                  std::string folder_prefix) {
    while (tex_elem) {
        std::string id = tex_elem->Attribute("id");
        TextureInfo info;
        const tinyxml2::XMLElement *element =
            tex_elem->FirstChildElement("string");
        while (element) {
            std::string path_value;
            if (parse_attribute(element, path_value, {"diffuse", "emission"})) {
                info.diff_path = folder_prefix + path_value;
            } else if (parse_attribute(element, path_value, {"specular"})) {
                info.spec_path = folder_prefix + path_value;
            } else if (parse_attribute(element, path_value,
                                       {"glossy", "sigma_a"})) {
                info.glos_path = folder_prefix + path_value;
            } else if (parse_attribute(element, path_value,
                                       {"rough1", "roughness_1", "ior"})) {
                info.rough_path1 = folder_prefix + path_value;
                info.is_rough_ior =
                    std::strcmp(element->Attribute("name"), "ior") == 0;
            } else if (parse_attribute(element, path_value,
                                       {"rough2", "roughness_2"})) {
                info.is_rough_ior = false;
                info.rough_path2 = folder_prefix + path_value;
            } else if (parse_attribute(element, path_value, {"normal"})) {
                info.normal_path = folder_prefix + path_value;
            } else {
                std::cerr << "Unsupported texture type '"
                          << element->Attribute("name") << "'\n";
                throw std::runtime_error("Unexpected texture type.");
            }
            element = element->NextSiblingElement("string");
        }
        texs.emplace(id, std::move(info));
        tex_elem = tex_elem->NextSiblingElement("texture");
    }
}

std::pair<PhaseFunction **, size_t>
parsePhaseFunction(const tinyxml2::XMLElement *phase_elem,
                   std::unordered_map<std::string, int> &phase_maps,
                   std::vector<TypedVec4> &phase_params) {
    const tinyxml2::XMLElement *traverser = phase_elem;
    size_t num_phase = 1; // allocate a dummy volume
    while (traverser) {
        ++num_phase;
        traverser = traverser->NextSiblingElement("phase");
    }

    PhaseFunction **phase_funcs = nullptr;
    phase_params.emplace_back();
    // We need at least one (Dummy), this won't cost much
    CUDA_CHECK_RETURN(
        cudaMalloc(&phase_funcs, sizeof(PhaseFunction *) * num_phase));
    CUDA_CHECK_RETURN(
        cudaMemset(phase_funcs, 0, sizeof(PhaseFunction *) * num_phase));
    create_device_phase<PhaseFunction><<<1, 1>>>(phase_funcs, 0);
    if (num_phase == 1) { // return when there is no valid phase function
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        return std::make_pair(phase_funcs, 0);
    }

    for (size_t i = 1; i < num_phase;
         i++, phase_elem = phase_elem->NextSiblingElement("phase")) {
        phase_params.emplace_back(Vec4(), PhaseFuncType::Isotropic);
        phase_maps[phase_elem->Attribute("id")] = i;
        std::string type = phase_elem->Attribute("type");
        type = type.empty() ? "hg" : type;

        if (type == "hg") {
            float g = 0.2;
            const tinyxml2::XMLElement *sub_elem =
                phase_elem->FirstChildElement("float");
            parse_attribute(sub_elem, g, {"g"});
            phase_params.back().first.x() = g;
            phase_params.back().second = PhaseFuncType::HenyeyGreenstein;
            create_device_phase<HenyeyGreensteinPhase>
                <<<1, 1>>>(phase_funcs, i, g);
        } else if (type == "isotropic") {
            create_device_phase<IsotropicPhase><<<1, 1>>>(phase_funcs, i);
        } else if (type == "hg-duo") {
            float g1 = 0.2, g2 = 0.8, weight = 0.5;
            const tinyxml2::XMLElement *sub_elem =
                phase_elem->FirstChildElement("float");
            while (sub_elem) {
                std::string name = sub_elem->Attribute("name");
                parse_attribute(sub_elem, g1, {"g1"});
                parse_attribute(sub_elem, g2, {"g2"});
                parse_attribute(sub_elem, weight, {"weight"});
                sub_elem = sub_elem->NextSiblingElement("string");
            }
            phase_params.back().first = Vec4(g1, g2, weight);
            phase_params.back().second = PhaseFuncType::DuoHG;
            create_device_phase<MixedHGPhaseFunction>
                <<<1, 1>>>(phase_funcs, i, g1, g2, weight);
        } else if (type == "rayleigh") {
            phase_params.back().second = PhaseFuncType::Rayleigh;
            create_device_phase<RayleighPhase><<<1, 1>>>(phase_funcs, i);
        } else if (type == "sggx") {
            std::cerr
                << "Current SGGX is not implemented but will be in the future. "
                   "Fall back to 'isotropic'\n";
            create_device_phase<IsotropicPhase><<<1, 1>>>(phase_funcs, i);
        } else {
            std::cerr << "Phase type '" << type
                      << "' not supported. Fall back to 'isotropic'\n";
            create_device_phase<IsotropicPhase><<<1, 1>>>(phase_funcs, i);
        }
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return std::make_pair(phase_funcs, num_phase - 1);
}

std::pair<Medium **, size_t>
parseMedium(const tinyxml2::XMLElement *vol_elem,
            const std::unordered_map<std::string, int> &phase_maps,
            const std::vector<TypedVec4> &phase_params,
            std::vector<MediumInfo> &med_infos,
            std::unordered_map<std::string, int> &medium_maps,
            GridVolumeManager &gvm,
            PhaseFunction **d_phase_funcs, // device_memory
            std::string folder_prefix) {
    const tinyxml2::XMLElement *traverser = vol_elem;
    size_t num_volume = 1; // allocate a dummy volume
    while (traverser) {
        ++num_volume;
        traverser = traverser->NextSiblingElement("medium");
    }

    // We need at least one (Dummy), this won't cost much
    Medium **d_volumes = nullptr;
    med_infos.emplace_back();
    CUDA_CHECK_RETURN(cudaMalloc(&d_volumes, sizeof(Medium *) * num_volume));
    create_device_medium<Medium><<<1, 1>>>(d_volumes, d_phase_funcs, 0, 0);
    if (num_volume == 1) {
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        return std::make_pair(d_volumes, 0);
    }

    for (size_t i = 1; i < num_volume;
         i++, vol_elem = vol_elem->NextSiblingElement("medium")) {
        std::string id = vol_elem->Attribute("id");
        std::string type = vol_elem->Attribute("type");
        medium_maps[id] = i;
        // phase function type
        size_t phase_id = 0;
        const tinyxml2::XMLElement *element =
            vol_elem->FirstChildElement("ref");
        if (element) {
            std::string ref_type = element->Attribute("type"),
                        ref_target = element->Attribute("id");
            if (ref_type == "phase") {
                auto it = phase_maps.find(ref_target);
                if (it != phase_maps.end()) {
                    phase_id = it->second;
                }
            }
        }
        if (phase_id == 0) {
            std::cerr << "Volume '" << id << "' has no valid phase function.\n";
            throw std::runtime_error("No valid phase function attached.");
        }
        MediumInfo med_info(id, element->Attribute("id"), phase_id,
                            MediumType::Homogeneous,
                            phase_params[phase_id].second);

        element = vol_elem->FirstChildElement("float");
        float scale = 1;
        while (element) {
            parse_attribute(element, scale, {"scale", "scaler"},
                            "Density scaler parsing error.");
            element = element->NextSiblingElement("float");
        }

        if (type == "homogeneous") {
            element = vol_elem->FirstChildElement("rgb");
            Vec4 sigma_a(0, 1), sigma_s(1, 1);
            while (element) {
                parse_attribute(element, sigma_a, {"sigma_a"});
                parse_attribute(element, sigma_s, {"sigma_s"});
                element = element->NextSiblingElement("rgb");
            }
            med_info.med_param = MediumInfo::MediumParams(
                sigma_a, sigma_s, phase_params[phase_id].first, scale);
            create_homogeneous_volume<<<1, 1>>>(
                d_volumes, d_phase_funcs, i, phase_id, sigma_a, sigma_s, scale);
        } else if (type == "grid") {
            med_info.mtype = MediumType::Grid;
            element = vol_elem->FirstChildElement("string");
            std::string name = element->Attribute("name"), density_path,
                        albedo_path = "", emission_path = "";
            while (element) {
                if (parse_attribute(element, density_path, {"density"})) {
                    density_path = folder_prefix + density_path;
                } else if (parse_attribute(element, albedo_path, {"albedo"})) {
                    albedo_path = folder_prefix + albedo_path;
                } else if (parse_attribute(element, emission_path,
                                           {"emission"})) {
                    emission_path = folder_prefix + emission_path;
                }
                element = element->NextSiblingElement("string");
            }

            element = vol_elem->FirstChildElement("rgb");
            Vec4 albedo(1, 1, 1);
            parse_attribute(element, albedo, {"albedo"});

            element = vol_elem->FirstChildElement("float");
            float temp_scale = 1, emission_scale = 1;
            while (element) {
                parse_attribute(element, temp_scale,
                                {"temp-scale", "temp_scale"});
                parse_attribute(element, emission_scale,
                                {"emission-scale", "em-scale"});
                element = element->NextSiblingElement("float");
            }

            if (albedo_path.empty()) {
                gvm.push(i, phase_id, density_path, albedo, scale, temp_scale,
                         emission_scale, emission_path);
            } else {
                gvm.push(i, phase_id, density_path, scale, temp_scale,
                         emission_scale, albedo_path, emission_path);
            }

            med_info.med_param = MediumInfo::MediumParams(
                albedo, Vec4(emission_scale, temp_scale, 0),
                phase_params[phase_id].first, scale);
        }
        med_infos.emplace_back(std::move(med_info));
    }
    if (!gvm.empty()) {
        gvm.to_gpu(d_volumes, d_phase_funcs);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return std::make_pair(d_volumes, num_volume - 1);
}

void parseObjectMediumRef(const tinyxml2::XMLElement *node,
                          std::unordered_map<std::string, int> &medium_map,
                          std::vector<int> &obj_med_idxs) {
    const tinyxml2::XMLElement *elem = node->FirstChildElement("ref");
    int idx = 0;
    while (elem) {
        std::string name = elem->Attribute("type");
        if (name == "medium") {
            auto it = medium_map.find(elem->Attribute("id"));
            if (it != medium_map.end()) {
                idx = it->second & 0x000000ff; // 8 bit, 255 media at most
            }
        }
        elem = elem->NextSiblingElement("ref");
    }
    if (!idx) {
        obj_med_idxs.push_back(0);
        return;
    }
    bool cullable = false;
    elem = node->FirstChildElement("bool");
    if (parse_attribute(elem, cullable, {"cullable"})) {
        // object_id (32bit): (31: is_sphere), (30: is_cullable), (29, 28
        // reserved), (27 - 20: medium idx)
        idx += static_cast<int>(cullable)
               << 10; // shift by 10 bit (then shift by 20, will be 30)
    }
    obj_med_idxs.push_back(idx);
}

const std::array<std::string, NumRendererType> RENDER_TYPE_STR = {
    "MegaKernel-PT",  "Wavefront-PT",     "Megakernel-LT",
    "Voxel-SDF-PT",   "Depth Tracer",     "BVH Cost Visualizer",
    "MegaKernel-VPT", "Accelerator-Only", "MegaKernel-PT (Dynamic)"};

Scene::Scene(std::string path)
    : num_bsdfs(0), num_emitters(0), num_objects(0), num_prims(0), envmap_id(0),
      cam_vol_id(0), num_phase_func(0), num_medium(0), media(nullptr),
      phases(nullptr) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) {
        std::cerr << "Failed to load file" << std::endl;
    }

    auto folder_prefix = get_folder_path(path);
    const tinyxml2::XMLElement
        *scene_elem = doc.FirstChildElement("scene"),
        *acc_elem = scene_elem->FirstChildElement("accelerator"),
        *bsdf_elem = scene_elem->FirstChildElement("brdf"),
        *shape_elem = scene_elem->FirstChildElement("shape"),
        *emitter_elem = scene_elem->FirstChildElement("emitter"),
        *sensor_elem = scene_elem->FirstChildElement("sensor"),
        *render_elem = scene_elem->FirstChildElement("renderer"),
        *texture_elem = scene_elem->FirstChildElement("texture"),
        *medium_elem = scene_elem->FirstChildElement("medium"),
        *phase_elem = scene_elem->FirstChildElement("phase"),
        *bool_elem = scene_elem->FirstChildElement("bool"), *ptr = nullptr;
    if (auto version_id = scene_elem->Attribute("version")) {
        if (std::strcmp(version_id, SCENE_VERSION) != 0) {
            std::cerr << "[SCENE] Version required: '" << SCENE_VERSION
                      << "', got '" << version_id << "'. Abort.\n";
            exit(0);
        }
    }

    std::unordered_map<std::string, int> bsdf_map, emitter_map, emitter_obj_map;
    std::vector<std::string> emitter_names;
    emitter_names.reserve(9);
    emitter_map.reserve(9);
    bsdf_map.reserve(48);

    // ------------------------- (0) parse the renderer
    // -------------------------
    std::string render_type =
        render_elem != nullptr ? render_elem->Attribute("type") : "pt";
    if (render_type == "pt")
        rdr_type = RendererType::MegaKernelPT;
    else if (render_type == "ptd")
        rdr_type = RendererType::MegaKernelPTDynamic;
    else if (render_type == "wfpt")
        rdr_type = RendererType::WavefrontPT;
    else if (render_type == "lt")
        rdr_type = RendererType::MegaKernelLT;
    else if (render_type == "sdf")
        rdr_type = RendererType::VoxelSDFPT;
    else if (render_type == "depth")
        rdr_type = RendererType::DepthTracing;
    else if (render_type == "vpt")
        rdr_type = RendererType::MegaKernelVPT;
    else if (render_type == "bvh-cost" || render_type == "bvh_cost")
        rdr_type = RendererType::BVHCostViz;
    else if (render_type == "accel-only" || render_type == "accel_only")
        rdr_type = RendererType::AcceleratorOnly;
    else {
        printf(
            "[Scene] Unknown renderer type: '%s', fall back to megakernel PT\n",
            render_type.c_str());
        rdr_type = RendererType::MegaKernelPT;
    }

    // ------------------------- (1) parse all the textures and BSDF
    // -------------------------

    std::unordered_map<std::string, TextureInfo> tex_map;
    std::unordered_map<std::string, int> phase_maps, medium_maps;
    std::vector<TypedVec4> phase_params;

    parseTexture(texture_elem, tex_map, folder_prefix);

    auto phase_pr = parsePhaseFunction(phase_elem, phase_maps, phase_params);
    phases = phase_pr.first;
    num_phase_func = phase_pr.second;

    auto media_pr =
        parseMedium(medium_elem, phase_maps, phase_params, medium_infos,
                    medium_maps, gvm, phases, folder_prefix);
    gvm.load_black_body_data(folder_prefix);
    media = media_pr.first;
    num_medium = media_pr.second;

    ptr = bsdf_elem;
    for (; ptr != nullptr; ++num_bsdfs)
        ptr = ptr->NextSiblingElement("brdf");
    if (num_bsdfs > MAX_ALLOWED_BSDF) {
        std::cerr << "Number of materials more than allowed. Max: "
                  << MAX_ALLOWED_BSDF << std::endl;
        throw std::runtime_error("Too many BSDF defined.");
    }
    CUDA_CHECK_RETURN(cudaMalloc(&bsdfs, sizeof(BSDF *) * num_bsdfs));
    CUDA_CHECK_RETURN(cudaMemset(bsdfs, 0, sizeof(BSDF *) * num_bsdfs));

    textures.init(num_bsdfs);
    for (int i = 0; i < num_bsdfs; i++) {
        parseBSDF(bsdf_elem, tex_map, bsdf_map, bsdf_infos, host_tex_4d,
                  host_tex_2d, textures, bsdfs, i);
        bsdf_elem = bsdf_elem->NextSiblingElement("brdf");
    }
    textures.to_gpu();
    CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(
        c_textures, &textures, sizeof(Textures), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // ------------------------- (2) parse emitter names
    // -------------------------
    parseEmitterNames(emitter_elem, emitter_map);

    // ------------------------- (3) parse all objects -------------------------
    ptr = shape_elem;
    for (; ptr != nullptr; ++num_objects)
        ptr = ptr->NextSiblingElement("shape");
    objects.reserve(num_objects);

    std::vector<bool> sphere_objs(num_objects, false);

    for (int i = 0; i < 3; i++) {
        verts_list[i].reserve(32);
        norms_list[i].reserve(32);
        uvs_list[i].reserve(32);
    }

    std::vector<int> obj_medium_idxs;
    obj_medium_idxs.reserve(num_objects);

    int prim_offset = 0;
    for (int i = 0; i < num_objects; i++) {
        std::string type = shape_elem->Attribute("type");
        if (type == "obj")
            parseObjShape(shape_elem, bsdf_map, emitter_map, bsdf_infos,
                          emitter_obj_map, objects, verts_list, norms_list,
                          uvs_list, prim_offset, folder_prefix, i);
        else if (type == "sphere")
            parseSphereShape(shape_elem, bsdf_map, emitter_map, bsdf_infos,
                             emitter_obj_map, objects, verts_list, norms_list,
                             uvs_list, prim_offset, folder_prefix, i);
        sphere_objs[i] = type == "sphere";

        parseObjectMediumRef(shape_elem, medium_maps, obj_medium_idxs);

        shape_elem = shape_elem->NextSiblingElement("shape");
    }

    num_prims = prim_offset;
    if (num_prims > MAX_PRIMITIVE_NUM) {
        // MAX_PRIMITIVE_NUM is the upper bound. 2^25 - 1, if num_prims exceeds
        // this bound For CompactNode, it is possible that the node offset will
        // be out-of-range
        std::cerr << "[Error] Too many primitives: " << num_prims
                  << " (maximum allowed: " << MAX_PRIMITIVE_NUM << ")\n";
        throw std::runtime_error("Too many primitives.");
    }

    //  ------------------------- (4) parse all emitters
    //  --------------------------
    ptr = emitter_elem;
    for (; ptr != nullptr; ++num_emitters)
        ptr = ptr->NextSiblingElement("emitter");
    CUDA_CHECK_RETURN(
        cudaMalloc(&emitters, sizeof(Emitter *) * (num_emitters + 1)));
    create_abstract_source<<<1, 1>>>(emitters[0]);
    for (int i = 1; i <= num_emitters; i++) {
        // FIXME: should light tracing support volumetric rendering, the medium
        // mapping should be passed in
        parseEmitter(emitter_elem, emitter_obj_map, tex_map, emitter_props,
                     emitter_names, host_tex_4d, emitters, envmap_id, i);
        emitter_elem = emitter_elem->NextSiblingElement("emitter");
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // ------------------------- (5) parse camera & scene config
    // -------------------------
    CUDA_CHECK_RETURN(cudaMallocHost(&cam, sizeof(DeviceCamera)));
    *cam = DeviceCamera::from_xml(sensor_elem);
    config = RenderingConfig::from_xml(acc_elem, render_elem, sensor_elem);

    // ------------------------- (6) initialize shapes -------------------------
    bool has_sphere = false;
    sphere_flags.resize(num_prims);
    prim_offset = 0;
    for (int obj_id = 0; obj_id < num_objects; obj_id++) {
        prim_offset += objects[obj_id].prim_num;
        bool is_sphere = sphere_objs[obj_id];
        for (int i = objects[obj_id].prim_offset; i < prim_offset; i++) {
            sphere_flags[i] = is_sphere;
            has_sphere |= is_sphere;
        }
    }

    printf("[BVH] Linear SAH-BVH is being built...\n");
    Vec3 world_min(AABB_INVALID_DIST), world_max(-AABB_INVALID_DIST);
    for (const auto &obj : objects) {
        obj.export_bound(world_min, world_max);
    }
    auto tp = std::chrono::system_clock::now();

#define CHRONO_OUTPUT(fmt_str, ...)                                            \
    {                                                                          \
        auto dur = std::chrono::system_clock::now() - tp;                      \
        auto count =                                                           \
            std::chrono::duration_cast<std::chrono::microseconds>(dur)         \
                .count();                                                      \
        auto elapsed = static_cast<double>(count) / 1e3;                       \
        printf(fmt_str, __VA_ARGS__);                                          \
    }

    if (has_sphere && config.bvh.use_sbvh) {
        config.bvh.use_sbvh = false;
        printf("[BVH] Primitives contain spheres. SBVH will not be enabled. "
               "Fall back to BVH.\n");
    }

    if (config.bvh.use_sbvh) {
        printf("[SBVH] Using spatial split during BVH building.\n");
        SBVHBuilder builder(verts_list, norms_list, uvs_list, sphere_flags,
                            objects, num_emitters, config.bvh.max_node_num);
        tp = std::chrono::system_clock::now();
        builder.build(obj_medium_idxs, world_min, world_max, obj_idxs, nodes,
                      cache_nodes, config.bvh.cache_level,
                      config.bvh.use_ref_unsplit);
        CHRONO_OUTPUT("[SBVH] BVH completed within %.3lf ms\n", elapsed);

        tp = std::chrono::system_clock::now();
        builder.post_process(obj_idxs, emitter_prims);

        CHRONO_OUTPUT(
            "[SBVH] Vertex data remapping completed within %.3lf ms\n",
            elapsed);
        int old_num_prims = num_prims;
        num_prims = verts_list[0].size();
        printf(
            "[SBVH] Primitives increased from %d to %d, increased by %.2f%%\n",
            old_num_prims, num_prims,
            float(num_prims - old_num_prims) / float(old_num_prims) * 100.f);
    } else {
        std::vector<int> prim_idxs; // won't need this if BVH is built
        BVHBuilder builder(verts_list, norms_list, uvs_list, sphere_flags,
                           objects, num_emitters, config.bvh.max_node_num,
                           config.bvh.bvh_overlap_w);
        tp = std::chrono::system_clock::now();
        builder.build(obj_medium_idxs, world_min, world_max, obj_idxs,
                      prim_idxs, nodes, cache_nodes, config.bvh.cache_level);
        CHRONO_OUTPUT("[BVH] BVH completed within %.3lf ms\n", elapsed);
        tp = std::chrono::system_clock::now();
        builder.post_process(prim_idxs, obj_idxs, emitter_prims);
        CHRONO_OUTPUT(
            "[BVH] Vertex data reordering completed within %.3lf ms\n",
            elapsed);
    }
    // The nodes.size is actually twice the number of nodes
    // since Each BVH node will be separated to two float4, nodes will store two
    // float4 for each node
}

Scene::~Scene() {
    destroy_gpu_alloc<<<1, num_bsdfs>>>(bsdfs);
    destroy_gpu_alloc<<<1, num_emitters + 1>>>(emitters);
    destroy_gpu_alloc<<<1, num_medium + 1>>>(media);
    destroy_gpu_alloc<<<1, num_phase_func + 1>>>(phases);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaFree(bsdfs));
    CUDA_CHECK_RETURN(cudaFree(emitters));
    CUDA_CHECK_RETURN(cudaFree(media));
    CUDA_CHECK_RETURN(cudaFree(phases));
    CUDA_CHECK_RETURN(cudaFreeHost(cam));
    for (auto &tex : host_tex_4d)
        tex.destroy();
    for (auto &tex : host_tex_2d)
        tex.destroy();
    textures.destroy();
}

CPT_KERNEL static void
vec2_to_packed_half_kernel(const Vec2 *src1, const Vec2 *src2, const Vec2 *src3,
                           PackedHalf2 *dst, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < count; i += blockDim.x * gridDim.x) {
        dst[i] = PackedHalf2(src1[i], src2[i], src3[i]);
    }
}

void Scene::update_emitters() {
    for (int index = 1; index <= num_emitters; index++) {
        Vec4 color = emitter_props[index - 1].second;
        if (color.x() < 0) {
            call_setter<<<1, 1>>>(emitters[index], color.y(),
                                  color.z() * DEG2RAD, color.w() * DEG2RAD);
        } else {
            set_emission<<<1, 1>>>(emitters[index], color.xyz(), color.w());
        }
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void Scene::update_materials() {
    for (size_t i = 0; i < bsdf_infos.size(); i++) {
        auto &bsdf_info = bsdf_infos[i];
        if (bsdf_info.bsdf_changed) {
            bsdf_info.bsdf_value_clamping();
            bsdf_info.create_on_gpu(bsdfs[i]);
        } else {
            bsdf_info.copy_to_gpu(bsdfs[i]);
        }
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void Scene::update_media() {
    if (rdr_type != RendererType::MegaKernelVPT)
        return;
    for (size_t i = 1; i < medium_infos.size(); i++) {
        auto &med_info = medium_infos[i];
        if (med_info.phase_changed) {
            med_info.clamp_phase_vals();
            med_info.create_on_gpu(media[i], phases);
        } else {
            med_info.clamp_phase_vals();
            med_info.copy_to_gpu(media[i], phases);
        }
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

template <typename T> static void free_resource(std::vector<T> &vec) {
    vec.clear();
    vec.shrink_to_fit();
}

void Scene::free_resources() {
    for (int i = 0; i < 3; i++) {
        free_resource(verts_list[i]);
        free_resource(norms_list[i]);
        free_resource(uvs_list[i]);
    }
    free_resource(objects);
    free_resource(sphere_flags);
    free_resource(obj_idxs);
    free_resource(nodes);
    free_resource(cache_nodes);
    free_resource(emitter_prims);
    gvm.free_resources();
}

void Scene::export_prims(PrecomputedArray &verts, NormalArray &norms,
                         ConstBuffer<PackedHalf2> &uvs) const {
    verts.from_vectors(verts_list[0], verts_list[1], verts_list[2],
                       &sphere_flags);
    norms.from_vectors(norms_list[0], norms_list[1], norms_list[2]);
    SoA3<Vec2> uvs_float(num_prims);
    uvs_float.from_vectors(uvs_list[0], uvs_list[1], uvs_list[2]);

    constexpr size_t block_size = 256;
    int num_blocks = (num_prims + block_size - 1) / block_size;
    vec2_to_packed_half_kernel<<<num_blocks, block_size>>>(
        &uvs_float.x(0), &uvs_float.y(0), &uvs_float.z(0), uvs.data(),
        num_prims);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    uvs_float.destroy();
}

void Scene::print() const noexcept {
    std::cout << " Rendering Settings:\n";
    std::cout << "\tRenderer type: " << RENDER_TYPE_STR[rdr_type] << std::endl;
    std::cout << "\t\tConfig: max depth:\t" << config.md.max_depth << std::endl;
    std::cout << "\t\tConfig: max diffuse:\t" << config.md.max_diffuse
              << std::endl;
    std::cout << "\t\tConfig: max specular:\t" << config.md.max_specular
              << std::endl;
    std::cout << "\t\tConfig: max transmit:\t" << config.md.max_tranmit
              << std::endl;
    std::cout << "\t\tConfig: Spec Cons:\t" << config.spec_constraint
              << std::endl;
    std::cout << "\t\tConfig: Bidirectional:\t" << config.bidirectional
              << std::endl;
    std::cout << "\t\tConfig: Caustics Scale:\t" << config.caustic_scaling
              << std::endl;
    std::cout << "\t\tConfig: SPP:\t\t" << config.spp << std::endl;
    std::cout << std::endl;

    std::cout << "\tAccelerator type: (S)BVH" << std::endl;
    std::cout << "\t\tSMem Cache Level: \t" << config.bvh.cache_level
              << std::endl;
    std::cout << "\t\tBVH Max Leaf Node: \t" << config.bvh.max_node_num
              << std::endl;
    std::cout << "\t\tBVH Overlap Weight: \t" << config.bvh.bvh_overlap_w
              << std::endl;
    std::cout << "\t\tBVH Spatial Split: \t" << config.bvh.use_sbvh
              << std::endl;
    std::cout << "\t\tReference Unsplit: \t";
    if (config.bvh.use_sbvh)
        std::cout << config.bvh.use_ref_unsplit << std::endl;
    else
        std::cout << "Not Applicable for non-SBVH" << std::endl;
    std::cout << std::endl;

    std::cout << "\tScene statistics: " << std::endl;
    std::cout << "\t\tNumber of objects: \t" << num_objects << std::endl;
    std::cout << "\t\tNumber of primitives: \t" << num_prims << std::endl;
    std::cout << "\t\tNumber of emitters: \t" << num_emitters << std::endl;
    std::cout << "\t\tNumber of BSDFs: \t" << num_bsdfs << std::endl;
    std::cout << std::endl;
    std::cout << "\tCamera Film Configs: " << std::endl;
    std::cout << "\t\tConfig: width:\t\t" << config.width << std::endl;
    std::cout << "\t\tConfig: height:\t\t" << config.height << std::endl;
    std::cout << "\t\tConfig: Gamma corr:\t" << config.gamma_correction
              << std::endl;
    std::cout << std::endl;

    if (config.md.max_time > 0) {
        std::cout << "\tToF statistics: " << std::endl;
        std::cout << "\t\tMin Time (unwarped):\t" << config.md.min_time
                  << std::endl;
        std::cout << "\t\tMax Time (unwarped):\t" << config.md.max_time
                  << std::endl;
    }
}
