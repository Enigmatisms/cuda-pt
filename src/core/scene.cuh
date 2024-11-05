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
#include "core/aos.cuh"
#include "core/bsdf.cuh"
#include "core/shapes.cuh"
#include "core/object.cuh"
#include "core/emitter.cuh"
#include "core/camera_model.cuh"
#include "core/virtual_funcs.cuh"
#include "core/bvh.cuh"

enum RendererType {
    MegaKernelPT = 0,            // megakernel path tracing
    WavefrontPT  = 1,            // wavefront  path tracing
    MegeKernelLT = 2,            // megakernel light tracing
    VoxelSDFPT   = 3,            // not supported currently
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
    std::vector<Shape> shapes;
    std::vector<bool> sphere_flags;
    std::vector<int> obj_idxs;
    std::vector<float4> nodes;
    std::vector<float4> cache_fronts;
    std::vector<float4> cache_backs;

    std::array<Vec3Arr, 3> verts_list;
    std::array<Vec3Arr, 3> norms_list;
    std::array<Vec2Arr, 3> uvs_list;
    std::vector<int> emitter_prims;

    RenderingConfig config;

    DeviceCamera* cam;
    int num_bsdfs;
    int num_prims;
    int num_emitters;
    int num_objects;
    const bool use_bvh;

    RendererType rdr_type;
public:
    Scene(std::string path);
    ~Scene();

    void export_prims(PrecomputedArray& verts, ArrayType<Vec3>& norms, ConstBuffer<PackedHalf2>& uvs) const;

    bool bvh_available() const noexcept {
        return !nodes.empty();
    }

    void print() const noexcept;
};