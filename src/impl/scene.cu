/**
 * Scene parser (from xml)
 * This is the implementation file
 * @author: Qianyue He
 * @date:   2024.9.6
*/
#include <numeric>
#include "core/scene.cuh"

const std::unordered_map<std::string, MetalType> conductor_mapping = {
    {"Au", MetalType::Au},
    {"Cr", MetalType::Cr},
    {"Cu", MetalType::Cu},
    {"Ag", MetalType::Ag},
    {"Al", MetalType::Al},
    {"W",   MetalType::W},
    {"TiO2", MetalType::TiO2},
    {"Ni",  MetalType::Ni},
    {"MgO", MetalType::MgO},
    {"Na",  MetalType::Na},
    {"SiC", MetalType::SiC},
    {"V",   MetalType::V},
    {"CuO", MetalType::CuO},
    {"Hg",  MetalType::Hg},
    {"Ir",  MetalType::Ir},
};

const std::unordered_map<std::string, DispersionType> dielectric_mapping = {
    {"Diamond",     DispersionType::Diamond},
    {"DiamondHigh", DispersionType::DiamondHigh},
    {"Silica",      DispersionType::Silica},
    {"Glass_BK7",   DispersionType::Glass_BK7},
    {"Glass_BaF10", DispersionType::Glass_BaF10},
    {"Glass_SF10",  DispersionType::Glass_SF10},
    {"Sapphire",    DispersionType::Sapphire},
    {"Water",       DispersionType::Water}
};

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

void parseBSDF(
    const tinyxml2::XMLElement* bsdf_elem, 
    const std::unordered_map<std::string, TextureInfo>& tex_map,
    std::unordered_map<std::string, int>& bsdf_map, 
    std::vector<BSDFInfo>& bsdf_infos,
    std::vector<Texture<float4>>& host_4d,
    std::vector<Texture<float2>>& host_2d,
    Textures& textures,
    BSDF** bsdfs, 
    int index
) {
    std::string type = bsdf_elem->Attribute("type");
    std::string id = bsdf_elem->Attribute("id");

    bsdf_map[id] = index;
    Vec4 k_d, k_s, k_g;

    const tinyxml2::XMLElement* element = bsdf_elem->FirstChildElement("rgb");
    while (element) {
        std::string name = element->Attribute("name");
        std::string value = element->Attribute("value");
        Vec4 color = parseColor(value);
        if (name == "k_d") {
            k_d = color;
        } else if (name == "k_s") {
            k_s = color;
        } else if (name == "k_g" || name == "sigma_a") {
            k_g = color;
        }
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
                std::cerr << "Texture named '" << value  << "' not found.\n";
                throw std::runtime_error("Referenced Texture not found.");
            } else {
                if (!it->second.diff_path.empty()) {
                    Texture<float4> tex(it->second.diff_path, TextureType::DIFFUSE_TEX);
                    textures.enqueue(tex, index);
                    host_4d.emplace_back(std::move(tex));
                }
                if (!it->second.spec_path.empty()) {
                    Texture<float4> tex(it->second.spec_path, TextureType::SPECULAR_TEX);
                    textures.enqueue(tex, index);
                    host_4d.emplace_back(std::move(tex));
                }
                if (!it->second.glos_path.empty()) {
                    Texture<float4> tex(it->second.glos_path, TextureType::GLOSSY_TEX);
                    textures.enqueue(tex, index);
                    host_4d.emplace_back(std::move(tex));
                }
                if (!it->second.rough_path1.empty()) {
                    Texture<float2> tex(
                        it->second.rough_path1, 
                        TextureType::ROUGHNESS_TEX, 
                        it->second.rough_path2, 
                        it->second.is_rough_ior
                    );
                    textures.enqueue(tex, index);
                    host_2d.emplace_back(std::move(tex));
                }
                if (!it->second.normal_path.empty()) {
                    Texture<float4> tex(it->second.normal_path, TextureType::NORMAL_TEX, "", false, true);
                    textures.enqueue(tex, index);
                    host_4d.emplace_back(std::move(tex));
                }
            }   
        }
    }

    BSDFInfo info(id);
    info.bsdf = BSDFInfo::BSDFParams(k_d, k_s, k_g);
    if (type == "lambertian") {
        create_bsdf<LambertianBSDF><<<1, 1>>>(bsdfs + index, k_d, k_s, k_g, BSDFFlag::BSDF_DIFFUSE | BSDFFlag::BSDF_REFLECT);
    } else if (type == "specular") {
        create_bsdf<SpecularBSDF><<<1, 1>>>(bsdfs + index, k_d, k_s, k_g, BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_REFLECT);
        info.type = BSDFType::Specular;
    } else if (type == "det-refraction") {
        create_bsdf<TranslucentBSDF><<<1, 1>>>(bsdfs + index, k_d, k_s, k_g, BSDFFlag::BSDF_SPECULAR | BSDFFlag::BSDF_TRANSMIT);
        info.type = BSDFType::Translucent;
    } else if (type == "conductor-ggx") {
        float roughness_x = 0.1f, roughness_y = 0.1f;
        MetalType mtype = MetalType::Cu;
        info.type = BSDFType::GGXConductor;
        element = bsdf_elem->FirstChildElement("string");
        if (element) {
            std::string name = element->Attribute("name");
            std::string value = element->Attribute("value");
            if (name == "type" || name == "metal" || name == "conductor") {
                std::string metal_type = element->Attribute("value");
                auto it = conductor_mapping.find(metal_type);
                if (it == conductor_mapping.end()) {
                    std::cout << "BSDF[" << id << "]" << ": Only << " << int(NumMetalType) << " types of metals are supported: ";
                    for (const auto [k, v]: conductor_mapping)
                        std::cout << k << ", ";
                    std::cout << std::endl;
                    std::cout << "Current type '" << metal_type << "' is not supported. Setting to 'Cu'\n";
                } else {
                    mtype = it->second;
                }
            }
        }
        element = bsdf_elem->FirstChildElement("float");
        tinyxml2::XMLError eResult;
        while (element) {
            std::string name = element->Attribute("name");
            std::string value = element->Attribute("value");
            if (name == "roughness_x" || name == "rough_x") {
                eResult = element->QueryFloatAttribute("value", &roughness_x);
                roughness_x = std::clamp(roughness_x, 0.001f, 1.f);
            } else if (name == "roughness_y" || name == "rough_y") {
                eResult = element->QueryFloatAttribute("value", &roughness_y);
                roughness_y = std::clamp(roughness_y, 0.001f, 1.f);
            }
            if (eResult != tinyxml2::XML_SUCCESS)
                throw std::runtime_error("Error parsing 'roughness' attribute");
            element = element->NextSiblingElement("float");
        }
        info.bsdf.store_ggx_params(mtype, k_g, roughness_x, roughness_y);
        create_metal_bsdf<<<1, 1>>>(bsdfs + index, METAL_ETA_TS[mtype], 
                    METAL_KS[mtype], k_g, roughness_x, roughness_y);
    } else if (type == "plastic" || type == "plastic-forward") {
        k_g = Vec4(0, 1);
        element = bsdf_elem->FirstChildElement("float");
        float trans_scaler = 1.f, thickness = 0.f, ior = 1.33f;
        while (element) {
            std::string name = element->Attribute("name");
            tinyxml2::XMLError eResult;
            if (name == "trans_scaler") {
                eResult = element->QueryFloatAttribute("value", &trans_scaler);
            } else if (name == "thickness") {
                eResult = element->QueryFloatAttribute("value", &thickness);
            } else if (name == "ior") {
                eResult = element->QueryFloatAttribute("value", &ior);
            }
            if (eResult != tinyxml2::XML_SUCCESS)
                throw std::runtime_error("Error parsing 'plastic BRDF' attribute");
            element = element->NextSiblingElement("float");
        }
        if (type == "plastic") {
            info.type = BSDFType::Plastic;
            create_plastic_bsdf<PlasticBSDF><<<1, 1>>>(bsdfs + index, 
                    k_d, k_s, k_g, ior, trans_scaler, thickness);
        } else {
            info.type = BSDFType::PlasticForward;
            create_plastic_bsdf<PlasticForwardBSDF><<<1, 1>>>(bsdfs + index, 
                    k_d, k_s, k_g, ior, trans_scaler, thickness);
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
                    std::cout << "BSDF[" << id << "]" << ": Only 8 types of metals are supported: ";
                    for (const auto [k, v]: dielectric_mapping)
                        std::cout << k << ", ";
                    std::cout << std::endl;
                    std::cout << "Current type '" << dielec_type << "' is not supported. Setting to 'Diamond'\n";
                } else {
                    dtype = it->second;
                }
            }
        }
        Vec2 dis_params = DISPERSION_PARAMS[dtype];
        info.type = BSDFType::Dispersion;
        info.bsdf.store_dispersion_params(dtype, k_s);
        create_dispersion_bsdf<<<1, 1>>>(bsdfs + index, k_s, dis_params.x(), dis_params.y());
    }
    bsdf_infos.emplace_back(std::move(info));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
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
    std::unordered_map<std::string, int>& emitter_obj_map,      // key emitter name, value object_id,
    const std::unordered_map<std::string, TextureInfo>& tex_map,
    std::vector<std::pair<std::string, Vec4>>& e_props,
    std::vector<std::string> obj_ref_names,
    std::vector<Texture<float4>>& host_texs,
    Emitter** emitters, 
    int& envmap_id,
    int index
) {
    std::string type = emitter_elem->Attribute("type");
    std::string id = emitter_elem->Attribute("id");

    obj_ref_names.push_back(id);
    Vec4 emission(0, 0, 0), scaler(1, 1, 1);

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
                std::cerr << "Texture named '" << value  << "' not found.\n";
                throw std::runtime_error("Referenced Texture not found.");
            } else {
                if (!it->second.diff_path.empty()) {
                    Texture<float4> tex(it->second.diff_path, TextureType::DIFFUSE_TEX);
                    tex_obj = tex.object();
                    host_texs.emplace_back(std::move(tex));
                } else {
                    std::cerr << "The texture for HDRI should be set for it's 'emission' element, but none is found.\n";
                    throw std::runtime_error("Referenced Texture not found.");
                }
            }
        }
    }

    if (type == "point") {
        element = emitter_elem->FirstChildElement("point");
        Vec3 pos(0, 0, 0);
        std::string name = element->Attribute("name");
        if (name == "center" || name == "pos")
            pos = parsePoint(element);
        create_point_source<<<1, 1>>>(emitters[index], emission * scaler, pos);
    } else if (type == "area-spot") {
        element = emitter_elem->FirstChildElement("float");
        float cos_val = 0.99;
        if (element) {
            std::string name = element->Attribute("name");
            tinyxml2::XMLError eResult;
            if (name == "half-angle" || name == "angle") {
                eResult = element->QueryFloatAttribute("value", &cos_val);
                cos_val = cosf(cos_val * DEG2RAD);
            }
            if (eResult != tinyxml2::XML_SUCCESS)
                throw std::runtime_error("Error parsing 'Area Spot Emitter' attribute");
        }
        element = emitter_elem->FirstChildElement("string");
        std::string attr_name = element->Attribute("name");
        if (!element || attr_name != "bind_type") {
            std::cerr << "Bound primitive is not specified for area spot source '" << id << "', name: "<< element->Attribute("name") << std::endl;
            throw std::runtime_error("Bound primitive is not specified for area spot source");
        }
        bool spherical_bound = element->Attribute("value") == std::string("sphere");
        create_area_spot_source<<<1, 1>>>(emitters[index], emission * scaler, cos_val, emitter_obj_map[id], spherical_bound, tex_obj);
    } else if (type == "area") {
        element = emitter_elem->FirstChildElement("string");
        std::string attr_name = element->Attribute("name");
        if (!element || attr_name != "bind_type") {
            std::cerr << "Bound primitive is not specified for area source '" << id << "', name: "<< element->Attribute("name") << std::endl;
            throw std::runtime_error("Bound primitive is not specified for area source");
        }
        bool spherical_bound = element->Attribute("value") == std::string("sphere");
        create_area_source<<<1, 1>>>(emitters[index], emission * scaler, emitter_obj_map[id], spherical_bound, tex_obj);
    } else if (type == "envmap") {
        envmap_id = index;
        element = emitter_elem->FirstChildElement("float");
        float scaler = 1.f, azimuth = 0.f, zenith = 0.f;
        while (element) {
            std::string name = element->Attribute("name");
            tinyxml2::XMLError eResult;
            if (name == "scaler") {
                eResult = element->QueryFloatAttribute("value", &scaler);
            } else if (name == "azimuth") {
                eResult = element->QueryFloatAttribute("value", &azimuth);
            } else if (name == "zenith") {
                eResult = element->QueryFloatAttribute("value", &zenith);
            }
            if (eResult != tinyxml2::XML_SUCCESS)
                throw std::runtime_error("Error parsing 'EnvMap Emitter' attribute");
            element = element->NextSiblingElement("float");
        }
        e_props.back().second = Vec4(-1, scaler, azimuth, zenith);
        element = emitter_elem->FirstChildElement("ref");
        if (tex_obj != 0) {
            create_envmap_source<<<1, 1>>>(emitters[index], 
                tex_obj, scaler, azimuth * DEG2RAD, zenith * DEG2RAD);
        } else {
            std::cerr << "Error: The texture for EnvMap is empty.\n";
            throw std::runtime_error("Referenced Texture not available.");
        }
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
    std::vector<BSDFInfo>& bsdf_infos,
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
    if (bsdf_id == -1) {
        std::cerr << "The current object does not have an attached BSDF.\n";
        throw std::runtime_error("Object with no BSDF");
    }
    bsdf_infos[bsdf_id].in_use = true;

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
    verts_list[1].emplace_back(radius, radius, radius);
    verts_list[2].emplace_back(0, 0, 0);

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
    std::vector<BSDFInfo>& bsdf_infos,
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
    if (bsdf_id == -1) {
        std::cerr << "The current object does not have an attached BSDF.\n";
        throw std::runtime_error("Object with no BSDF");
    }
    bsdf_infos[bsdf_id].in_use = true;

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
    ObjInfo object(bsdf_id, prim_offset, num_new_primitive, emitter_id);
    for (const auto& shape : shapes) {
        size_t num_primitives = shape.mesh.indices.size() / 3;
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
                    index = 2 * idx.texcoord_index;
                    uvs_list[j].emplace_back(attrib.texcoords[index], attrib.texcoords[index + 1]);
                } else {
                    uvs_list[j].emplace_back(0, 0);
                }
            }
            if (!has_normal) {      // compute normals ourselves
                printf("Normal vector not found in '%s' primitive %llu, computing yet normal direction is not guaranteed.\n", name.c_str(), i);
                Vec3 diff = verts_list[1][i] - verts_list[0][i];
                Vec3 normal = diff.cross(verts_list[2][i] - verts_list[0][i]).normalized_h();
                for (int j = 0; j < 3; j++) {
                    norms_list[j].push_back(normal);
                }
            }
        }

    }
    object.setup(verts_list);
    objects.push_back(object);
}

void parseTexture(
    const tinyxml2::XMLElement* tex_elem, 
    std::unordered_map<std::string, TextureInfo>& texs,
    std::string folder_prefix
) {
    while (tex_elem) {
        std::string id = tex_elem->Attribute("id");
        TextureInfo info;
        const tinyxml2::XMLElement* element = tex_elem->FirstChildElement("string");
        while (element) {
            std::string name = element->Attribute("name");
            if (name == "diffuse" || name == "emission") {
                info.diff_path = folder_prefix + element->Attribute("value");
            } else if (name == "specular") {
                info.spec_path = folder_prefix + element->Attribute("value");
            } else if (name == "glossy" || name == "sigma_a") {
                info.glos_path = folder_prefix + element->Attribute("value");
            } else if (name == "rough1" || name == "roughness_1" || name == "ior") {
                info.rough_path1 = folder_prefix + element->Attribute("value");
                info.is_rough_ior = name == "ior";
            } else if (name == "rough2" || name == "roughness_2") {
                info.is_rough_ior = false;
                info.rough_path2 = folder_prefix + element->Attribute("value");
            } else if (name == "normal") {
                info.normal_path = folder_prefix + element->Attribute("value");
            } else {
                std::cerr << "Unsupported texture type '" << name << "'\n";
                throw std::runtime_error("Unexpected texture type.");
            }
            element = element->NextSiblingElement("string");
        }
        texs.emplace(id, std::move(info));
        tex_elem = tex_elem->NextSiblingElement("texture");
    }
}

const std::array<std::string, NumRendererType> RENDER_TYPE_STR = {"MegaKernel-PT", "Wavefront-PT", "Megakernel-LT", "Voxel-SDF-PT"};

Scene::Scene(std::string path): num_bsdfs(0), num_emitters(0), num_objects(0), num_prims(0), envmap_id(0), use_bvh(false) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) {
        std::cerr << "Failed to load file" << std::endl;
    }

    auto folder_prefix = getFolderPath(path);
    const tinyxml2::XMLElement *scene_elem   = doc.FirstChildElement("scene"),
                                *bsdf_elem    = scene_elem->FirstChildElement("brdf"),
                                *shape_elem   = scene_elem->FirstChildElement("shape"),
                                *emitter_elem = scene_elem->FirstChildElement("emitter"),
                                *sensor_elem  = scene_elem->FirstChildElement("sensor"), 
                                *render_elem  = scene_elem->FirstChildElement("renderer"), 
                                *texture_elem = scene_elem->FirstChildElement("texture"), 
                                *bool_elem    = scene_elem->FirstChildElement("bool"), *ptr = nullptr;

    std::unordered_map<std::string, int> bsdf_map, emitter_map, emitter_obj_map;
    std::vector<std::string> emitter_names;
    emitter_names.reserve(9);
    emitter_map.reserve(9);
    bsdf_map.reserve(32);


    // ------------------------- (0) parse the renderer -------------------------
    std::string render_type = render_elem != nullptr ? render_elem->Attribute("type") : "pt";
    if      (render_type == "pt")   rdr_type = RendererType::MegaKernelPT;
    else if (render_type == "wfpt") rdr_type = RendererType::WavefrontPT;
    else if (render_type == "lt")   rdr_type = RendererType::MegeKernelLT;
    else if (render_type == "sdf")  rdr_type = RendererType::VoxelSDFPT;
    else                            rdr_type = RendererType::MegaKernelPT;
    
    {       // local field starts
    auto& use_bvh_ref = const_cast<bool&>(use_bvh);
    while (bool_elem) {
        std::string name = bool_elem->Attribute("name");
        std::string value = bool_elem->Attribute("value");
        if (name == "use_bvh") {
            std::transform(value.begin(), value.end(), value.begin(),
                    [](unsigned char c){ return std::tolower(c); });
            use_bvh_ref = value == "true";
        }
        bool_elem = bool_elem->NextSiblingElement("bool");
    }
    }       // local field ends

    // ------------------------- (1) parse all the textures and BSDF -------------------------
    
    std::unordered_map<std::string, TextureInfo> tex_map;
    parseTexture(texture_elem, tex_map, folder_prefix);

    ptr = bsdf_elem;
    for (; ptr != nullptr; ++ num_bsdfs)
        ptr = ptr->NextSiblingElement("brdf");
    CUDA_CHECK_RETURN(cudaMalloc(&bsdfs, sizeof(BSDF*) * num_bsdfs));

    textures.init(num_bsdfs);
    for (int i = 0; i < num_bsdfs; i++) {
        parseBSDF(bsdf_elem, tex_map, bsdf_map, bsdf_infos, host_tex_4d, host_tex_2d, textures, bsdfs, i);
        bsdf_elem = bsdf_elem->NextSiblingElement("brdf");
    }
    textures.to_gpu();
    CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(c_textures, &textures, sizeof(Textures), 0, cudaMemcpyHostToDevice));
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
            parseObjShape(shape_elem, bsdf_map, emitter_map, bsdf_infos, emitter_obj_map, 
                    objects, verts_list, norms_list, uvs_list, prim_offset, folder_prefix, i);
        else if (type == "sphere")
            parseSphereShape(shape_elem, bsdf_map, emitter_map, bsdf_infos, emitter_obj_map, 
                    objects, verts_list, norms_list, uvs_list, prim_offset, folder_prefix, i);
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
        parseEmitter(emitter_elem, emitter_obj_map, tex_map, emitter_props, emitter_names, host_tex_4d, emitters, envmap_id, i);
        emitter_elem = emitter_elem->NextSiblingElement("emitter");
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // ------------------------- (5) parse camera & scene config -------------------------
    CUDA_CHECK_RETURN(cudaMallocHost(&cam, sizeof(DeviceCamera)));
    *cam = DeviceCamera::from_xml(sensor_elem);
    config = RenderingConfig::from_xml(sensor_elem);

    // ------------------------- (6) initialize shapes -------------------------
    shapes.resize(num_prims);
    sphere_flags.resize(num_prims);
    prim_offset = 0;
    for (int obj_id = 0; obj_id < num_objects; obj_id ++) {
        prim_offset += objects[obj_id].prim_num;
        for (int i = objects[obj_id].prim_offset; i < prim_offset; i++) {
            if (!sphere_objs[obj_id]) {
                shapes[i] = TriangleShape(obj_id);
                sphere_flags[i] = false;
            } else {
                shapes[i] = SphereShape(obj_id);
                sphere_flags[i] = true;
            }
        }
    }

    if (use_bvh) {
        printf("[BVH] Linear SAH-BVH is being built...\n");
        Vec3 world_min(1e4, 1e4, 1e4), world_max(-1e4, -1e4, -1e4);
        for (const auto& obj: objects) {
            obj.export_bound(world_min, world_max);
        }
        auto tp = std::chrono::system_clock::now();
        std::vector<int> prim_idxs;     // won't need this if BVH is built
        bvh_build(
            verts_list[0], verts_list[1], verts_list[2], 
            objects, sphere_objs, world_min, world_max, 
            obj_idxs, prim_idxs, nodes, 
            cache_fronts, cache_backs, config.cache_level
        );
        auto dur = std::chrono::system_clock::now() - tp;
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
        auto elapsed = static_cast<double>(count) / 1e3;
        printf("[BVH] BVH completed within %.3lf ms\n", elapsed);
        // The nodes.size is actually twice the number of nodes
        // since Each BVH node will be separated to two float4, nodes will store two float4 for each node
        printf("[BVH] Total nodes: %llu, leaves: %llu\n", nodes.size() >> 1, prim_idxs.size());

        tp = std::chrono::system_clock::now();
        std::array<Vec3Arr, 3> reorder_verts, reorder_norms;
        std::array<Vec2Arr, 3> reorder_uvs;
        std::vector<bool> reorder_sph_flags(num_prims);

        for (int i = 0; i < 3; i++) {
            Vec3Arr &reorder_vs = reorder_verts[i],
                    &reorder_ns = reorder_norms[i];
            Vec2Arr &reorder_uv = reorder_uvs[i];
            const Vec3Arr &origin_vs = verts_list[i],
                          &origin_ns = norms_list[i];
            const Vec2Arr &origin_uv = uvs_list[i];
            reorder_vs.resize(num_prims);
            reorder_ns.resize(num_prims);
            reorder_uv.resize(num_prims);

            for (int j = 0; j < num_prims; j++) {
                int index = prim_idxs[j];
                reorder_vs[j] = origin_vs[index];
                reorder_ns[j] = origin_ns[index];
                reorder_uv[j] = origin_uv[index];
            }
        }

        // build an emitter primitive index map for emitter sampling
        // before the reordering logic, the emitter primitives are gauranteed
        // to be stored continuously, so we don't need an extra index map
        std::vector<std::vector<int>> eprim_idxs(num_emitters);
        std::vector<Shape> reorder_shapes(num_prims);
        for (int i = 0; i < num_prims; i++) {
            int index = prim_idxs[i], obj_idx = obj_idxs[i];
            obj_idx = obj_idx < 0 ? -obj_idx - 1 : obj_idx;
            const auto& object = objects[obj_idx];
            reorder_sph_flags[i] = sphere_flags[index];
            reorder_shapes[i] = shapes[index];
            if (object.is_emitter()) {
                int emitter_idx = object.emitter_id - 1;
                eprim_idxs[emitter_idx].push_back(i);
            }
        }
        // The following code does the following job:
        // BVH op will 'shuffle' the primitive order (sort of)
        // So, the emitter object might not have continuous
        // primitives stored in the memory. In order to uniformly sample
        // all the primitives on a given emitter, we should store the linearized
        // indices to the primitives, so the following code (1) linearize
        // the indices and (2) recalculate the object.prim_offset, while
        // the object.prim_cnt stays unchanged
        std::vector<int> e_prim_offsets;
        e_prim_offsets.push_back(0);
        for (const auto& eprim_idx: eprim_idxs) {
            e_prim_offsets.push_back(eprim_idx.size());
            for (int index: eprim_idx) 
                emitter_prims.push_back(index);
        }
        std::partial_sum(e_prim_offsets.begin(), e_prim_offsets.end(), e_prim_offsets.begin());
        for (ObjInfo& obj: objects) {
            if (!obj.is_emitter()) continue;
            obj.prim_offset = e_prim_offsets[obj.emitter_id - 1];
        }

        uvs_list     = std::move(reorder_uvs);
        verts_list   = std::move(reorder_verts);
        norms_list   = std::move(reorder_norms);
        sphere_flags = std::move(reorder_sph_flags);
        shapes       = std::move(reorder_shapes);
        dur = std::chrono::system_clock::now() - tp;
        count = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
        elapsed = static_cast<double>(count) / 1e3;
        printf("[BVH] Vertex data reordering completed within %.3lf ms\n", elapsed);
    } else {
        // To be consistent with the BVH branch, we store the primitive indices 
        prim_offset = 0;
        obj_idxs.clear();
        obj_idxs.reserve(num_prims);
        for (size_t i = 0; i < objects.size(); i++) {
            ObjInfo& obj = objects[i];
            if (obj.is_emitter()) {
                for (int cnt = 0; cnt < obj.prim_num; cnt++) {
                    obj_idxs.push_back(i);
                    emitter_prims.push_back(obj.prim_offset + cnt);
                }
                obj.prim_offset = prim_offset;
                prim_offset += obj.prim_num;
            } else {
                // if is not an emitter: just fill in the obj_idxs
                for (int cnt = 0; cnt < obj.prim_num; cnt++)
                    obj_idxs.push_back(i);
            }
        }
    }
}

Scene::~Scene() {
    destroy_gpu_alloc<<<1, num_bsdfs>>>(bsdfs);
    destroy_gpu_alloc<<<1, num_emitters + 1>>>(emitters);

    CUDA_CHECK_RETURN(cudaFree(bsdfs));
    CUDA_CHECK_RETURN(cudaFree(emitters));
    CUDA_CHECK_RETURN(cudaFreeHost(cam));
    for (auto& tex: host_tex_4d) tex.destroy();
    for (auto& tex: host_tex_2d) tex.destroy();
    textures.destroy();
}

CPT_KERNEL static void vec2_to_packed_half_kernel(const Vec2* src1, const Vec2* src2, const Vec2* src3, PackedHalf2* dst, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < count; i += blockDim.x * gridDim.x) {
        dst[i] = PackedHalf2(src1[i], src2[i], src3[i]);
    }
}

void Scene::update_emitters() {
    for (int index = 1; index <= num_emitters; index++) {
        Vec4 color = emitter_props[index - 1].second;
        if (color.x() < 0) {
            call_setter<<<1, 1>>>(emitters[index], color.y(), color.z() * DEG2RAD, color.w() * DEG2RAD);
        } else {
            set_emission<<<1, 1>>>(emitters[index], color.xyz(), color.w());
        }
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void Scene::update_materials() {
    for (size_t i = 0; i < bsdf_infos.size(); i++) {
        auto& bsdf_info = bsdf_infos[i];
        if (bsdf_info.bsdf_changed) {
            bsdf_info.bsdf_value_clamping();
            bsdf_info.create_on_gpu(bsdfs[i]);
        } else {
            bsdf_info.copy_to_gpu(bsdfs[i]);
        }
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

void Scene::export_prims(PrecomputedArray& verts, ArrayType<Vec3>& norms, ConstBuffer<PackedHalf2>& uvs) const {
    verts.from_vectors(verts_list[0], verts_list[1], verts_list[2], &sphere_flags);
    norms.from_vectors(norms_list[0], norms_list[1], norms_list[2]);
    ArrayType<Vec2> uvs_float(num_prims);
    uvs_float.from_vectors(uvs_list[0], uvs_list[1], uvs_list[2]);

    constexpr size_t block_size = 256;
    int num_blocks = (num_prims + block_size - 1) / block_size; // 计算所需 block 数
    vec2_to_packed_half_kernel<<<num_blocks, block_size>>>(&uvs_float.x(0), &uvs_float.y(0), &uvs_float.z(0), uvs.data(), num_prims);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    uvs_float.destroy();
}

void Scene::print() const noexcept {
    std::cout << " Scene:\n";
    std::cout << "\tRenderer type:\t\t" << RENDER_TYPE_STR[rdr_type] << std::endl;
    std::cout << "\tUse SAH-BVH:\t\t" << use_bvh << std::endl;
    std::cout << "\tSAH-BVH Cache Level: \t" << config.cache_level << std::endl;
    std::cout << "\tNumber of objects: \t" << num_objects << std::endl;
    std::cout << "\tNumber of primitives: \t" << num_prims << std::endl;
    std::cout << "\tNumber of emitters: \t" << num_emitters << std::endl;
    std::cout << "\tNumber of BSDFs: \t" << num_bsdfs << std::endl;
    std::cout << std::endl;
    std::cout << "\tConfig: width:\t\t" << config.width << std::endl;
    std::cout << "\tConfig: height:\t\t" << config.height << std::endl;
    std::cout << "\tConfig: max depth:\t" << config.md.max_depth << std::endl;
    std::cout << "\tConfig: max diffuse:\t" << config.md.max_diffuse << std::endl;
    std::cout << "\tConfig: max specular:\t" << config.md.max_specular << std::endl;
    std::cout << "\tConfig: max transmit:\t" << config.md.max_tranmit << std::endl;
    std::cout << "\tConfig: SPP:\t\t" << config.spp << std::endl;
    std::cout << "\tConfig: Gamma corr:\t" << config.gamma_correction << std::endl;
    std::cout << "\tConfig: Spec Cons:\t" << config.spec_constraint << std::endl;
    std::cout << "\tConfig: Bidirectional:\t" << config.bidirectional << std::endl;
    std::cout << "\tConfig: Caustics Scale:\t" << config.caustic_scaling << std::endl;
    std::cout << std::endl;
}