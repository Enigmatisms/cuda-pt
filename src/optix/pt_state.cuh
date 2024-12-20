/**
 * @file pt_state.cuh
 * @author Qianyue He
 * @brief Path tracer state
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <optix.h>
#include <optix_device.h>
#include <optix_stubs.h>
#include "core/ray.cuh"

#define OPTIX_CHECK(call) \
{\
    auto res = call;    \
    if (res != OPTIX_SUCCESS) { \
        std::cerr << "OptiX error at " << __FILE__ << ":" << __LINE__ << ": " << res << std::endl; \
        exit(1); \
    } \
}

struct RayOptiX {
    float3 origin;
    float3 direction;
    float tmin;
    float tmax;

    CPT_CPU_GPU RayOptiX(const Ray& ray, float t_min = 0, float t_max = MAX_DIST): 
        origin(ray.o), direction(ray.d), tmin(t_min), tmax(t_max) {}
};

struct ProgramGroup {
    OptixProgramGroup raygen_shader;
    OptixProgramGroup miss_shader;
    OptixProgramGroup ch_shader;        // closest hit shader
    OptixProgramGroup ah_shader;        // any hit shader

    ~ProgramGroup() {
        OPTIX_CHECK( optixProgramGroupDestroy( raygen_shader ) );
        OPTIX_CHECK( optixProgramGroupDestroy( miss_shader ) );
        OPTIX_CHECK( optixProgramGroupDestroy( ch_shader ) );
        OPTIX_CHECK( optixProgramGroupDestroy( ah_shader ) );
    }
};

struct PathTracerStates {
    OptixDeviceContext context;
    ProgramGroup pg;
    OptixPipelineCompileOptions compile_options;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    OptixTraversableHandle gas_handle;

    PathTracerStates(): context(0), pg{0, 0, 0}, compile_options{}, pipeline(0), sbt{} {
        CUcontext cuContext = 0;
        OPTIX_CHECK(optixDeviceContextCreate(cuContext, 0, &context));

        compile_options = {};
        compile_options.usesMotionBlur = false;
        compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        compile_options.numPayloadValues = 2;       // Payload 0: t, Payload 1: primitive_index
        compile_options.numAttributeValues = 2;     // u, v
        compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        compile_options.pipelineLaunchParamsVariableName = "params";
    }

    ~PathTracerStates() {
        OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
        OPTIX_CHECK( optixDeviceContextDestroy( context ) );
        CUDA_CHECK_RETURN( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord ) ) );
        CUDA_CHECK_RETURN( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase ) ) );
        CUDA_CHECK_RETURN( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
    }

    // copy operation is not allowed
    PathTracerStates(const PathTracerStates&) = delete;
    PathTracerStates& operator=(const PathTracerStates&) = delete;
};