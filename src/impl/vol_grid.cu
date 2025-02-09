#ifdef VALID_NANOVDB
#include "volume/grid.cuh"

CPT_KERNEL void create_grid_volume(
    const nanovdb::FloatGrid* dev_den_grids,
    const nanovdb::Vec3fGrid* dev_alb_grids,
    const nanovdb::FloatGrid* dev_ems_grids,
    Medium** media,
    float scale,
    PhaseFunction* ptr
) {
    if (threadIdx.x == 0) {
        *media = new GridVolumeMedium(dev_den_grids, dev_alb_grids, dev_ems_grids, scale);
        (*media)->bind_phase_function(ptr);
    }
};

CPT_CPU GridVolumeManager::GridVolumeManager() {
    host_handles.reserve(12);
    den_grids.reserve(4);
    alb_grids.reserve(4);
    ems_grids.reserve(4);
    phase_ptrs.reserve(4);
    medium_indices.reserve(4);
    const_albedos.reserve(4);
    scales.reserve(4);
}

// @overload, RGB albedo grid
CPT_CPU void GridVolumeManager::push(
    size_t          index, 
    std::string     den_path, 
    PhaseFunction*  ptr, 
    float           scale, 
    std::string     alb_path, 
    std::string     ems_path
) {
    medium_indices.push_back(index);
    scales.push_back(scale);
    phase_ptrs.push_back(ptr);
    from_vdb_file(den_path, den_grids);
    if (!from_vdb_file(alb_path, alb_grids)) {
        const_albedos.emplace_back(1, 1, 1);
    }
    from_vdb_file(ems_path, ems_grids);
}

    // @overload, inherently constant albedo grid
CPT_CPU void GridVolumeManager::push(
    size_t          index, 
    std::string     den_path, 
    Vec4            albedo, 
    PhaseFunction*  ptr, 
    float           scale, 
    std::string     ems_path
) {
    medium_indices.push_back(index);
    scales.push_back(scale);
    alb_grids.push_back(nullptr);
    phase_ptrs.push_back(ptr);
    const_albedos.emplace_back(std::move(albedo));
    from_vdb_file(ems_path, ems_grids);
}

CPT_CPU void GridVolumeManager::to_gpu(Medium** medium) {
    for (int i = 0; i < medium_indices.size(); i++) {
        size_t map_index = medium_indices[i];
        create_grid_volume<<<1, 1>>>(den_grids[i], alb_grids[i], ems_grids[i], medium + map_index, scales[i], phase_ptrs[i]);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

#endif  // VALID_NANOVDB