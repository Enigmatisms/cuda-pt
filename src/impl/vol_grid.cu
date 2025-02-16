#include "volume/grid.cuh"
#include "core/xyz.cuh"

static __constant__ cudaTextureObject_t emit_tex = 0;

CPT_GPU_INLINE float warp_reduce_float(float value, int start_mask = 16) {
    #pragma unroll
    for (int mask = start_mask; mask >= 1; mask >>= 1) {
        int float_int = __float_as_int(value);
        float_int = __shfl_xor_sync(0xffffffff, float_int, mask);
        value += __int_as_float(float_int);
    }
    return value;
}

CPT_GPU_INLINE float warp_reduce_int(int value, int start_mask = 16) {
    #pragma unroll
    for (int mask = start_mask; mask >= 1; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, mask);
    }
    return value;
}

// fast volume accumulation kernel blockDim: 1024
CPT_KERNEL void compute_volume_sum(
    const nanovdb::FloatGrid* density, 
    float* sum_avg, 
    int* valid_cell,
    int num_cells, 
    int width, int height,
    int x_s = 0, int y_s = 0, int z_s = 0
) {
    __shared__ float shared_sum[32];
    __shared__ int shared_sum_cnt[32];
    auto acc = density->getAccessor();
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int x = idx % width + x_s;
    int y = (idx / width) % height + y_s;
    int z = idx / (width * height) + z_s;

    float local = 0;
    if (idx < num_cells) {
        local = acc.getValue(nanovdb::Coord(x, y, z));
    }
    int local_cnt = local > 0;
    local     = warp_reduce_float(local);
    local_cnt = warp_reduce_int(local_cnt);
    if (threadIdx.x % 32 == 0) {
        shared_sum[threadIdx.x >> 5]     = local;
        shared_sum_cnt[threadIdx.x >> 5] = local_cnt;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        local     = warp_reduce_float(shared_sum[threadIdx.x]);
        local_cnt = warp_reduce_int(shared_sum_cnt[threadIdx.x]);
        if (threadIdx.x == 0) {
            atomicAdd(sum_avg, local);
            atomicAdd(valid_cell, local_cnt);
        }
    }
    __syncthreads();
}

CPT_GPU GridVolumeMedium::GridVolumeMedium(
    const nanovdb::FloatGrid* _den,
    float _avg_density,
    const nanovdb::Vec3fGrid* _alb,
    const nanovdb::FloatGrid* _em,
    Vec3 _const_alb,
    float _scale,
    float _temp_scale,
    float _em_scale
): density(_den), albedo(_alb), emission(_em), const_alb(std::move(_const_alb)), 
    scale(_scale), temp_scale(_temp_scale), emission_scale(_em_scale), avg_density(_avg_density) {
    auto bbox = density->worldBBox();
    grid_aabb = AABB(
        Vec3(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
        Vec3(bbox.max()[0], bbox.max()[1], bbox.max()[2]),
        0, 0
    );
}

CPT_GPU Vec4 GridVolumeMedium::query_emission(Vec3 pos, Sampler& sp) const {
    float temp = sample_temperature(std::move(pos), Vec3(sp.next1D()) - 0.5f);
    auto res = Vec4(tex1D<float4>(emit_tex, temp * temp_scale)) * emission_scale;
    return res;
}

CPT_GPU_INLINE MediumSample GridVolumeMedium::sample(const Ray& ray, Sampler& sp, float max_dist) const {
    float t_near = 0, t_far = 0;
    MediumSample ds{Vec4(1, 1), ray.hit_t, 0};
    if (grid_aabb.intersect(ray, t_near, t_far)) {
        t_near = fmaxf(0, t_near);
        t_far  = fminf(t_far, ray.hit_t);
        ds = delta_tracking_dist_sample(ray, sp, t_near, t_far);
    }
    return ds;
}

CPT_GPU_INLINE Vec4 GridVolumeMedium::transmittance(const Ray& ray, Sampler& sp, float dist) const {
    float Tr = 1, t_near = 0, t_far = 0;
    if (grid_aabb.intersect(ray, t_near, t_far)) {
        t_near = fmaxf(0, t_near);
        t_far  = fminf(t_far, ray.hit_t);
        Tr = ratio_tracking_trans_estimate(ray, sp, t_near, t_far);
    }   
    return Vec4(Tr, 1);
}

CPT_GPU MediumSample GridVolumeMedium::delta_tracking_dist_sample(
    const Ray& ray,
    Sampler& sp,
    float near_t,
    float far_t
) const { 
    float t, inv_maj;
    ((nanovdb::FloatTree*)density->treePtr())->extrema(t, inv_maj);
    inv_maj = 1.f / (scale * inv_maj);

    MediumSample ds{Vec4(1, 1), ray.hit_t, 0};
    t = near_t - logf(1.f - sp.next1D()) * inv_maj;
    while (t < far_t) {
        Vec3 sample_pos = ray.o + t * ray.d, offset = Vec3(sp.next1D()) - 0.5f;
        sample_pos = Vec3(sample_pos.x(), sample_pos.y(), sample_pos.z());
        const float d = sample_density(sample_pos, offset) * scale;
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

// TODO: residual tracking transmittance estimator
CPT_GPU_INLINE float GridVolumeMedium::residual_tracking_trans_estimate(
    const Ray& ray,
    Sampler& sp,
    float near_t,
    float far_t
) const {
    float Tr = 1.f, t = 0, inv_maj = 1;
    ((nanovdb::FloatTree*)density->treePtr())->extrema(t, inv_maj);
    // control u_c is set to the half of the majorant
    float sigma_c = scale * avg_density;
    inv_maj = 1.f / fmaxf(sigma_c, inv_maj * scale - sigma_c);

    t = near_t - logf(1.f - sp.next1D()) * inv_maj;
    while (t < far_t) {
        Vec3 sample_pos = ray.o + t * ray.d, offset = Vec3(sp.next1D()) - 0.5f;
        sample_pos = Vec3(sample_pos.x(), sample_pos.y(), sample_pos.z());
        const float d = sample_density(sample_pos, offset) * scale;
        Tr *= 1.f - (d - sigma_c) * inv_maj;
        if (Tr < 0.1f) {            // Russian Roulette
            if (sp.next1D() >= Tr) return 0.f;
            Tr = 1.f;
        }
        t -= logf(1.f - sp.next1D()) * inv_maj;
    }
    return Tr * expf(-sigma_c * (far_t - near_t));
}

CPT_GPU float GridVolumeMedium::ratio_tracking_trans_estimate(
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
        Vec3 sample_pos = ray.o + t * ray.d, offset = Vec3(sp.next1D()) - 0.5f;
        sample_pos = Vec3(sample_pos.x(), sample_pos.y(), sample_pos.z());
        const float d = sample_density(sample_pos, offset) * scale;
        Tr *= fmaxf(0.f, 1.f - d * inv_maj);

        if (Tr < 0.1f) {            // Russian Roulette
            if (sp.next1D() >= Tr) return 0.f;
            Tr = 1.f;
        }
        t -= logf(1.f - sp.next1D()) * inv_maj;
    }
    return Tr;
}

CPT_KERNEL void create_grid_volume(
    const nanovdb::FloatGrid* dev_den_grids,
    const nanovdb::Vec3fGrid* dev_alb_grids,
    const nanovdb::FloatGrid* dev_ems_grids,
    Medium** media,
    PhaseFunction** phases,
    int med_id, int ph_id,
    Vec3 _calb,
    float scale,
    float temp_scale,
    float em_scale,
    float avg_density           // for residual ratio tracking
) {
    if (threadIdx.x == 0) {
        media[med_id] = new GridVolumeMedium(dev_den_grids, avg_density, dev_alb_grids, 
            dev_ems_grids, std::move(_calb), scale, temp_scale, em_scale);
        media[med_id]->bind_phase_function(phases[ph_id]);
    }
};

CPT_CPU GridVolumeManager::GridVolumeManager(): _emit_tex(0) {
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
    float           temp_scale, 
    float           em_scale, 
    std::string     alb_path, 
    std::string     ems_path
) {
    medium_indices.push_back(med_id);
    phase_indices.push_back(ph_id);
    scales.push_back(scale);
    temp_scales.push_back(temp_scale);
    em_scales.push_back(em_scale);
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
    float           temp_scale, 
    float           em_scale, 
    std::string     ems_path
) {
    medium_indices.push_back(med_id);
    phase_indices.push_back(ph_id);
    scales.push_back(scale);
    temp_scales.push_back(temp_scale);
    em_scales.push_back(em_scale);
    from_vdb_file(den_path, den_grids);
    alb_grids.push_back(nullptr);
    const_albedos.emplace_back(Vec3(albedo.x(), albedo.y(), albedo.z()));
    from_vdb_file(ems_path, ems_grids);
}

CPT_CPU void GridVolumeManager::to_gpu(Medium** medium, PhaseFunction** phases) {
    for (int i = 0; i < medium_indices.size(); i++) {
        size_t media_idx = medium_indices[i],
               phase_idx = phase_indices[i];
        const auto& grid_handle = host_handles[i];
        auto bbox = grid_handle.grid<float>()->indexBBox();
        auto bbox_min = bbox.min(), bbox_diff = bbox.max() - bbox_min;
        int num_cells = (bbox_diff.x() + 1) * (bbox_diff.y() + 1) * (bbox_diff.z() + 1);

        // calculate average density for residual ratio tracking
        float *sum_val = nullptr, host_val = 0;
        int   *cnt_val = nullptr, host_cnt = 0;
        CUDA_CHECK_RETURN(cudaMalloc(&sum_val, sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc(&cnt_val, sizeof(int)));
        CUDA_CHECK_RETURN(cudaMemset(sum_val, 0, sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemset(cnt_val, 0, sizeof(int)));
        compute_volume_sum<<<(num_cells + 1023) >> 10, 1024>>>(den_grids[i], 
            sum_val, cnt_val, num_cells, bbox_diff.x() + 1, bbox_diff.y() + 1, bbox_min.x(), bbox_min.y(), bbox_min.z());
        CUDA_CHECK_RETURN(cudaMemcpy(&host_val, sum_val, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(&host_cnt, cnt_val, sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK_RETURN(cudaFree(sum_val));
        CUDA_CHECK_RETURN(cudaFree(cnt_val));

        create_grid_volume<<<1, 1>>>(den_grids[i], alb_grids[i], 
            ems_grids[i], medium, phases, media_idx, phase_idx, 
            const_albedos[i], scales[i], temp_scales[i], em_scales[i], host_val / float(host_cnt));
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

CPT_CPU GridVolumeManager::~GridVolumeManager() {
    if (_emit_tex) {
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(_emit_tex));
        CUDA_CHECK_RETURN(cudaFreeArray(_bb_emission));
    }
}

CPT_CPU void GridVolumeManager::load_black_body_data(std::string path_prefix) {
    std::ifstream file(path_prefix + "../data/blackbody.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open binary file containing blackbody emission data." << std::endl;
        throw std::runtime_error("File read failed.");
    }
    std::vector<float4> data;
    while (file) {
        float3 point;
        file.read(reinterpret_cast<char*>(&point), sizeof(float3));
        if (file) {
            data.push_back({point.x, point.y, point.z, 1.f});
        }
    }
    file.close();
    printf("[VOLUME] Blackbody emission data loaded.\n");
    _emit_tex = createArrayTexture1D<float4>(data.data(), _bb_emission, data.size());
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(emit_tex, &_emit_tex, sizeof(cudaTextureObject_t)));
}