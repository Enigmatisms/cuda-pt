#pragma once
/**
 * @file homogeneous.cuh
 * @author Qianyue He
 * @brief Grid volume
 * @version 0.1
 * @date 2025-02-05
 * @copyright Copyright (c) 2025
 */

#include <vector>
#include "core/medium.cuh"
#include <nanovdb/io/IO.h>
#include <nanovdb/cuda/GridHandle.cuh>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/NodeManager.cuh>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>

class GridVolumeMedium: public Medium {
public:
    // GridVolumeMedium does not directly control the life-span
    // of the grid data, so it just holds the pointer to the 
    // device grid
    const nanovdb::FloatGrid* density;      // density
    const nanovdb::Vec3fGrid* albedo;       // albedo Vec3f
    const nanovdb::FloatGrid* emission;     // emission grid: blackbody temperature
    AABB grid_aabb;
    Vec3 const_alb;                         // if `albedo` is not nullptr, this field is useless
    float scale;                            // scale of the density
    float temp_scale;                       // scale the max temperature
    float emission_scale;                   // scale of the emission
    const float avg_density;

    // currently, volume traversal is sampled via nearest neighbor
    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU_INLINE float sample_density(VType1&& pos, VType2&& offset) const {
        nanovdb::Vec3f idx = density->worldToIndexF(nanovdb::Vec3f(pos.x(), pos.y(), pos.z()));
        idx += nanovdb::Vec3f(offset.x(), offset.y(), offset.z());
        auto acc = density->getAccessor();
        return acc.getValue(nanovdb::Coord(roundf(idx[0]), roundf(idx[1]), roundf(idx[2])));
    }

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU_INLINE float sample_temperature(VType1&& pos, VType2&& offset) const {
        if (emission == nullptr) return 0;
        nanovdb::Vec3f idx = emission->worldToIndexF(nanovdb::Vec3f(pos.x(), pos.y(), pos.z()));
        idx += nanovdb::Vec3f(offset.x(), offset.y(), offset.z());
        auto acc = emission->getAccessor();
        return acc.getValue(nanovdb::Coord(roundf(idx[0]), roundf(idx[1]), roundf(idx[2])));
    }

    CONDITION_TEMPLATE_SEP_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU_INLINE Vec4 sample_albedo(VType1&& pos, VType2&& offset) const {
        if (albedo == nullptr) return Vec4(const_alb.x(), const_alb.y(), const_alb.z());
        nanovdb::Vec3f idx = albedo->worldToIndexF(nanovdb::Vec3f(pos.x(), pos.y(), pos.z()));
        idx += nanovdb::Vec3f(offset.x(), offset.y(), offset.z());
        auto acc = albedo->getAccessor();
        nanovdb::Vec3f val = acc.getValue(nanovdb::Coord(roundf(idx[0]), roundf(idx[1]), roundf(idx[2])));
        return Vec4(val[0], val[1], val[2]);
    }

    CPT_GPU GridVolumeMedium(
        const nanovdb::FloatGrid* _den,
        float _avg_density,
        const nanovdb::Vec3fGrid* _alb = nullptr,
        const nanovdb::FloatGrid* _em = nullptr,
        Vec3 _const_alb = Vec3(1), 
        float _scale = 1,
        float _temp_scale = 1,
        float _em_scale   = 1
    );

    CPT_GPU MediumSample delta_tracking_dist_sample(
        const Ray& ray,
        Sampler& sp,
        float near_t,
        float far_t
    ) const;

    CPT_GPU float ratio_tracking_trans_estimate(
        const Ray& ray,
        Sampler& sp,
        float near_t,
        float far_t
    ) const;

    // TODO: residual tracking transmittance estimator
    CPT_GPU float residual_tracking_trans_estimate(
        const Ray& ray,
        Sampler& sp,
        float near_t,
        float far_t
    ) const;

    CPT_GPU_INLINE MediumSample sample(const Ray& ray, Sampler& sp, float max_dist = MAX_DIST) const override;

    CPT_GPU_INLINE Vec4 transmittance(const Ray& ray, Sampler& sp, float dist) const override;

    CPT_GPU Vec4 query_emission(Vec3 pos, Sampler& sp) const override;

    CONDITION_TEMPLATE(VType, Vec4)
    CPT_GPU_INLINE void set_params(VType&& _const_alb, float _s, float _tp_s, float _em_s) {
        const_alb      = Vec3(_const_alb.x(), _const_alb.y(), _const_alb.z());
        scale          = _s;
        temp_scale     = _tp_s;
        emission_scale = _em_s;
    }
};

class GridVolumeManager {
private:
    template <typename DType>
    using GridType = nanovdb::Grid<nanovdb::NanoTree<DType>>;

    std::vector<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>> host_handles;

    std::vector<nanovdb::FloatGrid*> den_grids;
    std::vector<nanovdb::Vec3fGrid*> alb_grids;
    std::vector<nanovdb::FloatGrid*> ems_grids;

    // from all medium to the grid volume medium, for example:
    // [homogeneous, homogeneous, grid, grid, homogeneous, grid] -> [2, 3, 5]
    std::vector<size_t> medium_indices;
    std::vector<size_t> phase_indices;
    std::vector<Vec3> const_albedos;    // if the grid volume has a constant albedo, then set here
    std::vector<float> scales;          // density scale
    std::vector<float> temp_scales;     // temporature scale
    std::vector<float> em_scales;       // emission scale

    cudaArray_t _bb_emission;
    cudaTextureObject_t _emit_tex;
private:
    template <typename DType>
    CPT_CPU bool from_vdb_file(std::string path, std::vector<GridType<DType>*>& dev_ptr_buffer) {
        if (!path.empty()) {
            auto handle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(path);
            handle.deviceUpload();
            host_handles.push_back(std::move(handle));
            dev_ptr_buffer.push_back(host_handles.back().deviceGrid<DType>());
            return true;
        } else {
            dev_ptr_buffer.push_back(nullptr);
            return false;
        }
    }
public:
    CPT_CPU GridVolumeManager();
    CPT_CPU ~GridVolumeManager();
    // @overload, RGB albedo grid
    CPT_CPU void push(
        size_t          med_id, 
        size_t          ph_id, 
        std::string     den_path, 
        float           scale = 1.f, 
        float           temp_scale = 1.f, 
        float           em_scale = 1.f, 
        std::string     alb_path = "", 
        std::string     ems_path = ""
    );
    // @overload, inherently constant albedo grid
    CPT_CPU void push(
        size_t          med_id, 
        size_t          ph_id, 
        std::string     den_path, 
        Vec4            albedo, 
        float           scale = 1.f, 
        float           temp_scale = 1.f, 
        float           em_scale = 1.f, 
        std::string     ems_path = ""
    );

    CPT_CPU void to_gpu(Medium** medium, PhaseFunction** phases);

    CPT_CPU bool empty() const noexcept {
        return host_handles.empty();
    }

    CPT_CPU void load_black_body_data(std::string path_prefix);

    CPT_CPU void free_resources();
};