/**
 * @file sbt.cuh
 * @author Qianyue He
 * @brief Shader Binding Table
 * @version 0.1
 * @date 2024-12-19
 * @copyright Copyright (c) 2024
 */

#pragma once
#include "optix/pt_state.cuh"

struct RaygenRecord {
    __align__(16) OptixProgramGroup prog_group;
};

struct MissRecord {
    __align__(16) OptixProgramGroup prog_group;
};

struct HitGroupRecord {
    __align__(16) OptixProgramGroup prog_group;
};

// create and load SBT to device side
inline void create_sbt(
    PathTracerStates& states
) {
    RaygenRecord rg_record = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( states.pg.raygen_shader, &rg_record ) );

    MissRecord miss_record = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( states.pg.miss_shader, &miss_record ) );

    HitGroupRecord ch_record = {}, ah_record = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( states.pg.ch_shader, &ch_record ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( states.pg.ah_shader, &ah_record ) );

    HitGroupRecord hitgroup_records[2] = { ch_record, ah_record };
    RaygenRecord* d_raygen_records;
    CUDA_CHECK_RETURN(cudaMalloc(&d_raygen_records, sizeof(RaygenRecord)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_raygen_records, &rg_record, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    MissRecord* d_miss_records;
    CUDA_CHECK_RETURN(cudaMalloc(&d_miss_records, sizeof(MissRecord)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_miss_records, &miss_record, sizeof(MissRecord), cudaMemcpyHostToDevice));

    HitGroupRecord* d_hit_records;
    CUDA_CHECK_RETURN(cudaMalloc(&d_hit_records, 2 * sizeof(HitGroupRecord)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_hit_records, &hitgroup_records, 2 * sizeof(HitGroupRecord), cudaMemcpyHostToDevice));

    states.sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_records);
    states.sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(d_miss_records);
    states.sbt.missRecordStrideInBytes = sizeof(MissRecord);
    states.sbt.missRecordCount = 1;
    states.sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_hit_records);
    states.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    states.sbt.hitgroupRecordCount = 2; // Closest Hit and Any Hit
}