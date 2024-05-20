/**
 * CUDA object information
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/bsdf.cuh"
#include "core/shapes.cuh"

class Object {
private:
    AABB _aabb;
public:
    int obj_index;          // index pf the current object
    int prim_offset;        // offset to the start of the primitives
    int prim_num;           // number of primitives
    uint8_t emitter_id;     // index to the emitter, 0xff means not an emitter
public:
    CPT_CPU_GPU_INLINE bool intersect(const Ray& ray, float& t_near) const noexcept {
        return _aabb.intersect(ray, t_near);
    }

    CPT_CPU_GPU Object() {}
};