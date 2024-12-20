#include <optix.h>
#include <optix_stubs.h>
#include <iostream>

/**
 * This is actually pretty fucked.
 * The following header must be included to avoid 
 * 'undefined symbol g_optixFunctionTable'
 * Check this out: https://forums.developer.nvidia.com/t/undefined-symbol-g-optixfunctiontable-during-execution/274690
 */
#include <optix_function_table_definition.h>

int main() {
    // 初始化 OptiX
    OptixDeviceContext context;
    OptixResult result = optixInit();
    if (result != OPTIX_SUCCESS) {
        std::cerr << "OptiX initialization failed" << std::endl;
        return -1;
    }
    return 0;
}