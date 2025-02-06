#pragma once
/**
 * @file homogeneous.cuh
 * @author Qianyue He
 * @brief Grid volume
 * @version 0.1
 * @date 2025-02-05
 * @copyright Copyright (c) 2025
 */

#ifdef VALID_NANOVDB
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
    float scale;

    // currently, volume traversal is sampled via nearest neighbor
    CONDITION_TEMPLATE_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU_INLINE float sample_density(VType1&& pos, VType2&& offset) const {
        nanovdb::Vec3f idx = density->worldToIndexF(nanovdb::Vec3f(pos.x(), pos.y(), pos.z()));
        idx += nanovdb::Vec3f(offset.x(), offset.y(), offset.z());
        auto acc = density->getAccessor();
        return acc.getValue(nanovdb::Coord::Floor(idx));
    }

    CONDITION_TEMPLATE_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU_INLINE float sample_temperature(VType1&& pos, VType2&& offset) const {
        nanovdb::Vec3f idx = emission->worldToIndexF(nanovdb::Vec3f(pos.x(), pos.y(), pos.z()));
        idx += nanovdb::Vec3f(offset.x(), offset.y(), offset.z());
        auto acc = emission->getAccessor();
        return acc.getValue(nanovdb::Coord::Floor(idx));
    }

    CONDITION_TEMPLATE_2(VType1, VType2, Vec3, Vec3)
    CPT_GPU_INLINE Vec4 sample_albedo(VType1&& pos, VType2&& offset) const {
        if (albedo == nullptr) return Vec4(const_alb.x(), const_alb.y(), const_alb.z());
        nanovdb::Vec3f idx = albedo->worldToIndexF(nanovdb::Vec3f(pos.x(), pos.y(), pos.z()));
        idx += nanovdb::Vec3f(offset.x(), offset.y(), offset.z());
        auto acc = albedo->getAccessor();
        nanovdb::Vec3f val = acc.getValue(nanovdb::Coord::Floor(idx));
        return Vec4(val[0], val[1], val[2]);
    }

    CPT_GPU GridVolumeMedium(
        const nanovdb::FloatGrid* _den,
        const nanovdb::Vec3fGrid* _alb = nullptr,
        const nanovdb::FloatGrid* _em = nullptr,
        float _scale = 1
    ): density(_den), albedo(_alb), emission(_em), scale(_scale) {
        auto bbox = density->worldBBox();
        grid_aabb = AABB(
            Vec3(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
            Vec3(bbox.max()[0], bbox.max()[1], bbox.max()[2]),
            0, 0
        );
    }

    CPT_GPU_INLINE MediumSample delta_tracking_dist_sample(
        const Ray& ray,
        Sampler& sp,
        float near_t,
        float far_t
    ) const { 
        float t, inv_maj;
        ((nanovdb::FloatTree*)density->treePtr())->extrema(t, inv_maj);
        inv_maj = 1.f / (scale * inv_maj);

        MediumSample ds{Vec4(1, 1), ray.hit_t, 1, 0};
        t = near_t - logf(1.f - sp.next1D()) * inv_maj;
        while (t < far_t) {
            Vec3 sample_pos = ray.o + t * ray.d, offset = Vec3(sp.next1D(), sp.next1D(), sp.next1D()) - 0.5f;
            const float d = sample_density(sample_pos, offset);
            if (sp.next1D() < d * inv_maj) {        // real collision
                ds.dist = t;
                ds.local_thp = sample_albedo(std::move(sample_pos), std::move(offset));
                ds.flag = 1;
                break;
            }
            t -= logf(1.f - sp.next1D()) * inv_maj;
        }
        return ds;
    }

    CPT_GPU_INLINE float ratio_tracking_trans_estimate(
        const Ray& ray,
        Sampler& sp,
        float near_t,
        float far_t
    ) const {
        float Tr = 1.f, t = 0, inv_maj = 1;
        ((nanovdb::FloatTree*)density->treePtr())->extrema(t, inv_maj);
        inv_maj = 1.f / (scale * inv_maj);

        t = near_t - logf(1.f - sp.next1D()) * inv_maj;
        while (t < far_t) {
            Vec3 sample_pos = ray.o + t * ray.d, offset = Vec3(sp.next1D(), sp.next1D(), sp.next1D()) - 0.5f;
            const float d = sample_density(sample_pos, offset);
            Tr *= max(0.f, 1.f - d * inv_maj);

            if (Tr < 0.1f) {            // Russian Roulette
                if (sp.next1D() >= Tr) return 0.f;
                Tr = 1.f;
            }
            t -= logf(1.f - sp.next1D()) * inv_maj;
        }
        return Tr;
    }

    // TODO: residual tracking transmittance estimator
    CPT_GPU_INLINE float residual_tracking_trans_estimate(
        const Ray& ray,
        Sampler& sp,
        float near_t,
        float far_t
    ) const {
        // Not implemented yet
        return 0.f;         
    }

    CPT_GPU_INLINE MediumSample sample(const Ray& ray, Sampler& sp, float max_dist = MAX_DIST) const override {
        float t_near = 0, t_far = 0;
        MediumSample ds{Vec4(1, 1), ray.hit_t, 1, 0};
        if (grid_aabb.intersect(ray, t_near, t_far)) {
            ds = delta_tracking_dist_sample(ray, sp, t_near, t_far);
        }
        return ds;
    }

    CPT_GPU_INLINE Vec4 transmittance(const Ray& ray, Sampler& sp, float dist) const override {
        float t_near = 0, t_far = 0, Tr = 1;
        if (grid_aabb.intersect(ray, t_near, t_far)) {
            Tr = ratio_tracking_trans_estimate(ray, sp, t_near, t_far);
        }
        return Vec4(Tr, 1);
    } 
};

CPT_KERNEL void create_grid_volume(
    const nanovdb::FloatGrid** dev_den_grids,
    const nanovdb::Vec3fGrid** dev_alb_grids,
    const nanovdb::FloatGrid** dev_ems_grids,
    Medium** media,
    int volume_index,
    float scale
) {
    if (threadIdx.x == 0) {
        *media = new GridVolumeMedium(dev_den_grids[volume_index], dev_alb_grids[volume_index], dev_ems_grids[volume_index], scale);
    }
};

class HostGridVolumes {
private:
    template <typename DType>
    using GridType = nanovdb::Grid<nanovdb::NanoTree<DType>>;

    std::vector<nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>> host_handles;

    std::vector<nanovdb::FloatGrid*> den_grids;
    std::vector<nanovdb::Vec3fGrid*> alb_grids;
    std::vector<nanovdb::FloatGrid*> ems_grids;

    nanovdb::FloatGrid** dev_den_grids;
    nanovdb::Vec3fGrid** dev_alb_grids;
    nanovdb::FloatGrid** dev_ems_grids;

    // from all medium to the grid volume medium, for example:
    // [homogeneous, homogeneous, grid, grid, homogeneous, grid] -> [2, 3, 5]
    std::vector<int> medium_indices;
    std::vector<Vec3> const_albedos;    // if the grid volume has a constant albedo, then set here
    std::vector<float> scales;          // density scale
public:
    CPT_CPU HostGridVolumes(): dev_den_grids(nullptr), dev_alb_grids(nullptr), dev_ems_grids(nullptr) {
        host_handles.reserve(12);
        den_grids.reserve(4);
        alb_grids.reserve(4);
        ems_grids.reserve(4);
        medium_indices.reserve(4);
        const_albedos.reserve(4);
        scales.reserve(4);
    }

    // @overload, RGB albedo grid
    CPT_CPU void push(int index, std::string den_path, float scale = 1.f, std::string alb_path = "", std::string ems_path = "") {
        medium_indices.push_back(index);
        scales.push_back(scale);

        from_vdb_file(den_path, den_grids);
        if (!from_vdb_file(alb_path, alb_grids)) {
            const_albedos.emplace_back(1, 1, 1);
        }
        from_vdb_file(ems_path, ems_grids);
    }

    // @overload, inherently constant albedo grid
    CPT_CPU void push(int index, std::string den_path, Vec3 albedo, float scale = 1.f, std::string ems_path = "") {
        medium_indices.push_back(index);
        scales.push_back(scale);
        alb_grids.push_back(nullptr);
        const_albedos.emplace_back(std::move(albedo));
        from_vdb_file(ems_path, ems_grids);
    }

    template <typename DType>
    CPT_CPU bool from_vdb_file(std::string path, std::vector<GridType<DType>*>& dev_ptr_buffer) {
        if (!path.empty()) {
            host_handles.push_back(nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(path));
            dev_ptr_buffer.push_back(host_handles.back().deviceGrid<DType>());
            return true;
        } else {
            dev_ptr_buffer.push_back(nullptr);
            return false;
        }
    }

    CPT_CPU ~HostGridVolumes() {
        if (dev_den_grids) CUDA_CHECK_RETURN(cudaFree(dev_den_grids));
        if (dev_alb_grids) CUDA_CHECK_RETURN(cudaFree(dev_alb_grids));
        if (dev_ems_grids) CUDA_CHECK_RETURN(cudaFree(dev_ems_grids));
    }

    CPT_CPU void to_gpu(Medium** medium) {
        CUDA_CHECK_RETURN(cudaMalloc(&dev_den_grids, sizeof(nanovdb::FloatGrid*) * den_grids.size()));
        CUDA_CHECK_RETURN(cudaMalloc(&dev_alb_grids, sizeof(nanovdb::Vec3fGrid*) * alb_grids.size()));
        CUDA_CHECK_RETURN(cudaMalloc(&dev_ems_grids, sizeof(nanovdb::FloatGrid*) * ems_grids.size()));

        CUDA_CHECK_RETURN(cudaMemcpy(dev_den_grids, den_grids.data(), 
                    sizeof(nanovdb::FloatGrid*) * den_grids.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_alb_grids, alb_grids.data(), 
                    sizeof(nanovdb::Vec3fGrid*) * alb_grids.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_ems_grids, ems_grids.data(), 
                    sizeof(nanovdb::FloatGrid*) * ems_grids.size(), cudaMemcpyHostToDevice));
        for (int i = 0; i < medium_indices.size(); i++) {
            int map_index = medium_indices[i];
            create_grid_volume<<<1, 1>>>(dev_den_grids, dev_alb_grids, dev_ems_grids, medium + map_index, i, scales[i]);
        }
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    }
};

#endif  // VALID_NANOVDB