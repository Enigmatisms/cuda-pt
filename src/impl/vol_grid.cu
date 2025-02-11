#include "volume/grid.cuh"

CPT_KERNEL void create_grid_volume(
    const nanovdb::FloatGrid* dev_den_grids,
    const nanovdb::Vec3fGrid* dev_alb_grids,
    const nanovdb::FloatGrid* dev_ems_grids,
    Medium** media,
    PhaseFunction** phases,
    int med_id, int ph_id,
    float scale
) {
    if (threadIdx.x == 0) {
        media[med_id] = new GridVolumeMedium(dev_den_grids, dev_alb_grids, dev_ems_grids, scale);
        media[med_id]->bind_phase_function(phases[ph_id]);
    }
};

CPT_CPU GridVolumeManager::GridVolumeManager() {
    host_handles.reserve(12);
    den_grids.reserve(4);
    alb_grids.reserve(4);
    ems_grids.reserve(4);
    phase_indices.reserve(4);
    medium_indices.reserve(4);
    const_albedos.reserve(4);
    scales.reserve(4);
}

// @overload, RGB albedo grid
CPT_CPU void GridVolumeManager::push(
    size_t          med_id, 
    size_t          ph_id, 
    std::string     den_path, 
    float           scale, 
    std::string     alb_path, 
    std::string     ems_path
) {
    medium_indices.push_back(med_id);
    phase_indices.push_back(ph_id);
    scales.push_back(scale);
    from_vdb_file(den_path, den_grids);
    if (!from_vdb_file(alb_path, alb_grids)) {
        const_albedos.emplace_back(1, 1, 1);
    }
    from_vdb_file(ems_path, ems_grids);
}

    // @overload, inherently constant albedo grid
CPT_CPU void GridVolumeManager::push(
    size_t          med_id, 
    size_t          ph_id, 
    std::string     den_path, 
    Vec4            albedo, 
    float           scale, 
    std::string     ems_path
) {
    medium_indices.push_back(med_id);
    phase_indices.push_back(ph_id);
    scales.push_back(scale);
    alb_grids.push_back(nullptr);
    const_albedos.emplace_back(std::move(albedo));
    from_vdb_file(ems_path, ems_grids);
}

CPT_CPU void GridVolumeManager::to_gpu(Medium** medium, PhaseFunction** phases) {
    for (int i = 0; i < medium_indices.size(); i++) {
        size_t media_idx = medium_indices[i],
               phase_idx = phase_indices[i];
        create_grid_volume<<<1, 1>>>(den_grids[i], alb_grids[i], 
            ems_grids[i], medium, phases, media_idx, phase_idx, scales[i]);
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}