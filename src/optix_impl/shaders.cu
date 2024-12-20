// raygen.cu
#include <optix.h>
#include <optix_device.h>

// Ray Generation Program
extern "C" __device__ void __raygen__rg() {
    // empty, currently, megakernel PT itself generates the rays
}

// Miss Program
extern "C" __device__ void __miss__ms() {
    optixSetPayload_0(0);
}

// Closest Hit Program
extern "C" __device__ void __closesthit__ch() {
    float t = optixGetRayTmax();
    unsigned int primitive_index = optixGetPrimitiveIndex();

    float u = optixGetTriangleBarycentrics().x;
    float v = optixGetTriangleBarycentrics().y;

    optixSetPayload_0(__float_as_uint(t));          // Payload 0: hit distance
    optixSetPayload_1(primitive_index);             // Payload 1: primitive index
    optixSetPayload_2(__float_as_uint(u));          // Payload 2: barycentric coordinate u
    optixSetPayload_3(__float_as_uint(v));          // Payload 3: barycentric coordinate v
}

// Any Hit Program
extern "C" __device__ void __anyhit__ah() {
    optixSetPayload_0(1); // Payload 0: occlusion flag, 1 means occluded
    // stop tracing
    optixTerminateRay();
}
