/**
 * CUDA object information
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/bsdf.cuh"
#include "core/shapes.cuh"

class ObjInfo {
private:
    AABB _aabb;
public:
    int bsdf_id;            // index pf the current object
    int prim_offset;        // offset to the start of the primitives
    int prim_num;           // number of primitives
    uint8_t emitter_id;     // index to the emitter, 0xff means not an emitter
public:
    CPT_CPU_GPU_INLINE bool intersect(const Ray& ray, float& t_near) const noexcept {
        return _aabb.intersect(ray, t_near);
    }

    CPT_CPU_GPU_INLINE bool is_emitter() const noexcept { return this->emitter_id > 0; }
    CPT_CPU_GPU ObjInfo(int bsdf_id, int prim_off, int prim_num, uint8_t emitter_id = 0):
        bsdf_id(bsdf_id), prim_offset(prim_off), prim_num(prim_num), emitter_id(emitter_id)
    {}
};