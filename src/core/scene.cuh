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
 * @date:   2024.5.24
 */
#pragma once
#include "bsdf/bsdf.cuh"
#include "core/bvh_spatial.cuh"
#include "core/camera_model.cuh"
#include "core/config.h"
#include "core/dynamic_bsdf.cuh"
#include "core/emitter.cuh"
#include "core/object.cuh"
#include "core/textures.cuh"
#include "core/virtual_funcs.cuh"
#include "volume/medium_registry.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <tiny_obj_loader.h>
#include <tinyxml2.h>
#include <unordered_map>
#include <vector>

using Vec4Arr = std::vector<Vec4>;
using Vec3Arr = std::vector<Vec3>;
using Vec2Arr = std::vector<Vec2>;

extern const std::array<std::string, NumRendererType> RENDER_TYPE_STR;

class Scene {
  public:
    BSDF **bsdfs;
    Emitter **emitters;
    PhaseFunction **phases;
    Medium **media;
    std::vector<ObjInfo> objects;
    std::vector<bool> sphere_flags;
    std::vector<int> obj_idxs;
    std::vector<float4> nodes;
    std::vector<CompactNode> cache_nodes;

    std::array<Vec3Arr, 3> verts_list;
    std::array<Vec3Arr, 3> norms_list;
    std::array<Vec2Arr, 3> uvs_list;
    std::vector<int> emitter_prims;

    // texture related
    Textures textures;
    std::vector<Texture<float4>> host_tex_4d;
    std::vector<Texture<float2>> host_tex_2d;
    // used in Scene property online update, now we can
    // modify the emitter emission on-the-fly
    std::vector<std::pair<std::string, Vec4>> emitter_props;
    std::vector<BSDFInfo> bsdf_infos;
    std::vector<MediumInfo> medium_infos;
    GridVolumeManager gvm;

    RenderingConfig config;

    DeviceCamera *cam;

    int num_bsdfs;
    int num_prims;
    int num_emitters;
    int num_objects;
    int envmap_id;
    int cam_vol_id;
    int num_phase_func;
    int num_medium;

    RendererType rdr_type;

  public:
    Scene(std::string path);
    ~Scene();

    void export_prims(PrecomputedArray &verts, NormalArray &norms,
                      ConstBuffer<PackedHalf2> &uvs) const;
    void free_resources();

    bool bvh_available() const noexcept { return !nodes.empty(); }

    void update_emitters();
    void update_materials();
    void update_media();

    void print() const noexcept;
};
