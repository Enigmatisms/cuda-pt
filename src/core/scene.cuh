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
#include "core/config.h"
#include "bsdf/bsdf.cuh"
#include "core/object.cuh"
#include "core/emitter.cuh"
#include "core/textures.cuh"
#include "core/camera_model.cuh"
#include "core/virtual_funcs.cuh"
#include "core/dynamic_bsdf.cuh"
#include "core/bvh.cuh"

enum RendererType: uint8_t {
    MegaKernelPT = 0,            // megakernel path tracing
    WavefrontPT  = 1,            // wavefront  path tracing
    MegeKernelLT = 2,            // megakernel light tracing
    VoxelSDFPT   = 3,            // not supported currently
    DepthTracing = 4,            // rendering depth map
    BVHCostViz   = 5,            // displaying BVH traversal cost
    NumRendererType
};

using Vec4Arr = std::vector<Vec4>;
using Vec3Arr = std::vector<Vec3>;
using Vec2Arr = std::vector<Vec2>;

extern const std::array<std::string, NumRendererType> RENDER_TYPE_STR;

class Scene {
public:
    BSDF** bsdfs;
    Emitter** emitters;
    std::vector<ObjInfo> objects;
    std::vector<bool> sphere_flags;
    std::vector<int> obj_idxs;
    std::vector<CompactNode> nodes;
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

    RenderingConfig config;

    DeviceCamera* cam;
    int num_bsdfs;
    int num_prims;
    int num_emitters;
    int num_objects;
    int envmap_id;
    const bool use_bvh;

    RendererType rdr_type;
public:
    Scene(std::string path);
    ~Scene();

    void export_prims(PrecomputedArray& verts, NormalArray& norms, ConstBuffer<PackedHalf2>& uvs) const;
    void free_resources();

    bool bvh_available() const noexcept {
        return !nodes.empty();
    }

    void update_emitters();
    void update_materials();

    void print() const noexcept;
};