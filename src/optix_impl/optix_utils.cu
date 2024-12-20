#include <array>
#include <cuda.h>
#include <fstream>
#include <sstream>
#include <optix_stack_size.h>
#include "core/vec3.cuh"
#include "optix/optix_utils.cuh"

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

void createContext( PathTracerStates& state )
{
    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );
    state.context = context;
}

std::string load_ptx_shader(const std::string& filename) {
    std::ifstream ptx_file(filename);
    if (!ptx_file.is_open()) {
        std::cerr << "Failed to open PTX file: " << filename << std::endl;
        exit(1);
    }
    std::stringstream ptxStream;
    ptxStream << ptx_file.rdbuf();
    return ptxStream.str();
}

float4* upload_vertices(const std::array<std::vector<Vec3>, 3>& verts_list, size_t num_prims) {
    const size_t num_verts = num_prims * 3;
    std::vector<float4> optix_vertices(num_verts);
    for (size_t i = 0; i < num_verts; i += 3) {
        float3 v1 = verts_list[0][i],
               v2 = verts_list[1][i],
               v3 = verts_list[2][i];
        optix_vertices[i]     = make_float4(v1.x, v1.y, v1.z, 0);
        optix_vertices[i + 1] = make_float4(v2.x, v2.y, v2.z, 0);
        optix_vertices[i + 2] = make_float4(v2.x, v2.y, v2.z, 0);
    }

    float4* d_vertices;
    CUDA_CHECK_RETURN(cudaMalloc(&d_vertices, sizeof(float4) * num_verts));
    CUDA_CHECK_RETURN(cudaMemcpy(d_vertices, optix_vertices.data(), sizeof(float4) * num_verts, cudaMemcpyHostToDevice));
    return d_vertices;
}

void build_accel(PathTracerStates& states, float4* d_vertices, size_t num_prims) {
    OptixBuildInput build_input;
    memset(&build_input, 0, sizeof(build_input));

    // triangle input
    OptixBuildInputTriangleArray& triangles = build_input.triangleArray;

    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangles.vertexStrideInBytes = sizeof(float4);
    triangles.numVertices = num_prims * 3;
    static CUdeviceptr v_buffers_host[1];
    v_buffers_host[0] = reinterpret_cast<CUdeviceptr>(d_vertices);
    triangles.vertexBuffers = v_buffers_host;

    // no index buffer, store all the vertices in a linear buffer
    triangles.indexBuffer = 0;
    triangles.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    triangles.numIndexTriplets = 0;

    // acceleration structure params
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // temp memory buffer
    OptixAccelBufferSizes buffer_sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(states.context, &accel_options, &build_input, 1, &buffer_sizes));

    CUdeviceptr d_temp_buffer;
    CUdeviceptr d_output_buffer;
    CUDA_CHECK_RETURN(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), buffer_sizes.tempSizeInBytes));
    CUDA_CHECK_RETURN(cudaMalloc(reinterpret_cast<void**>(&d_output_buffer), buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        states.context,
        /* CUDA Stream */ 0,
        &accel_options,
        &build_input,
        1,
        d_temp_buffer,
        buffer_sizes.tempSizeInBytes,
        d_output_buffer,
        buffer_sizes.outputSizeInBytes,
        &states.gas_handle,
        /* emittedProperties */ nullptr,
        /* numEmittedProperties */ 0
    ));

    CUDA_CHECK_RETURN(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
}

void create_program_group(PathTracerStates& states, const std::string& ptxFile, bool verbose) {
    std::string ptx_code = load_ptx_shader(ptxFile);

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixModule ptx_module = nullptr;
    
    char log[2048];
    size_t log_size = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(
        states.context,
        &module_compile_options,
        &states.compile_options, 
        ptx_code.c_str(),
        ptx_code.size(),
        log, &log_size,
        &ptx_module
    ));

    if (verbose && log_size > 1) {
        std::cout << "[OptiX] Module creation log: " << log << std::endl;
    }

    OptixProgramGroupOptions program_group_options = {};

    // Ray Gen Program Group
    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK(optixProgramGroupCreate(
            states.context, &raygen_prog_group_desc,
            1,
            &program_group_options,
            nullptr, nullptr,
            &states.pg.raygen_shader
        ));
    }

    // Miss Program Group
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

        OPTIX_CHECK(optixProgramGroupCreate(
            states.context, &miss_prog_group_desc,
            1,
            &program_group_options,
            nullptr, nullptr,
            &states.pg.miss_shader
        ));
    }

    // closest hit shader Program Group
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = ptx_module; // Closest Hit
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        hit_prog_group_desc.hitgroup.moduleAH = nullptr;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

        OPTIX_CHECK(optixProgramGroupCreate(
            states.context,
            &hit_prog_group_desc,
            1,
            &program_group_options,
            nullptr, nullptr,
            &states.pg.ch_shader
        ));
    }

    {   // any hit shader
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = nullptr;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        hit_prog_group_desc.hitgroup.moduleAH = ptx_module; // Any Hit
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";

        OPTIX_CHECK(optixProgramGroupCreate(
            states.context,
            &hit_prog_group_desc,
            1,
            &program_group_options,
            nullptr, nullptr,
            &states.pg.ah_shader
        ));
    }
    
    OPTIX_CHECK(optixModuleDestroy(ptx_module));
}

void create_pipeline(PathTracerStates& states)
{
    // 定义 Hit Groups：Path Hit Group 和 Shadow Hit Group
    OptixProgramGroup program_groups[] =
    {
        states.pg.ch_shader,        // Shading Hit Group (only CH)
        states.pg.ah_shader,        // Shadow Hit Group (only AH)
        states.pg.miss_shader       // Miss Shader
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;        // non-recursive

    OPTIX_CHECK(optixPipelineCreate(
        states.context,
        &states.compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        nullptr, nullptr,
        &states.pipeline
    ));

    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(states.pg.ch_shader, &stack_sizes, states.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(states.pg.ah_shader, &stack_sizes, states.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(states.pg.miss_shader, &stack_sizes, states.pipeline));

    constexpr uint32_t max_trace_depth = 1;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;

    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    
    OPTIX_CHECK(optixPipelineSetStackSize(
        states.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_trace_depth
    ));
}