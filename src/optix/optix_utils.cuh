#pragma once

#include "optix/pt_state.cuh"
/**
 * This is actually pretty fucked.
 * The following header must be included to avoid 
 * 'undefined symbol g_optixFunctionTable'
 * Check this out: https://forums.developer.nvidia.com/t/undefined-symbol-g-optixfunctiontable-during-execution/274690
 */
#include <optix_function_table_definition.h>

void create_pipeline(PathTracerStates& states);

std::string load_ptx_shader(const std::string& filename);

/**
 * @note it is said in OptiX documentation that padding float3 to float4 (16 byte stride) 
 * is better for GAS building. That's what we will do here.
 */
float4* upload_vertices(const std::array<std::vector<Vec3>, 3>& verts_list, size_t num_prims);

// create optix acceleration structure, returns traversable_handle
void build_accel(
    PathTracerStates& states,
    float4* d_vertices, 
    size_t num_prims
);

// create shader group
void create_program_group(PathTracerStates& states, const std::string& ptxFile, bool verbose);
