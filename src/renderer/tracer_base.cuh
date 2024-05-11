/**
 * Simple tile-based depth renderer
 * @date: 5.12.2024
 * @author: Qianyue He
*/

#include <variant/variant.h>
#include "core/soa.cuh"
#include "core/shapes.cuh"
#include "core/camera_model.cuh"
#include "core/host_device.cuh"
#include "core/stats.h"

extern __constant__ DeviceCamera dev_cam;

namespace {
    using Shape = variant::variant<SphereShape, TriangleShape>;
    using ConstShapePtr = const Shape* const;
}

class TracerBase {
public:
    __global__ void render(
        ConstShapePtr shapes,
        ConstAABBPtr aabbs,
        ConstPrimPtr verts,
        ConstPrimPtr norms, 
        ConstUVPtr uvs,
        DeviceImage& image,
        int num_prims,
        int max_depth = 1/* max depth, useless for depth renderer, 1 anyway */
    ) { throw std::runtime_error("Not implemented."); }
};