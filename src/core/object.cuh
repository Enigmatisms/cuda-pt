/**
 * CUDA object information
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include <array>
#include "core/aos.cuh"
#include "core/aabb.cuh"

class ObjInfo {
private:
    AABB _aabb;
public:
    int bsdf_id;            // index pf the current object
    int prim_offset;        // offset to the start of the primitives
    int prim_num;           // number of primitives
    float inv_area;         // inverse area
    uint8_t emitter_id;     // index to the emitter, 0 means not an emitter
public:
    CPT_GPU_INLINE bool intersect(const Ray& ray, float& t_near) const noexcept {
        return _aabb.intersect(ray, t_near);
    }

    CPT_CPU_GPU void setup(const ArrayType<Vec3>& prims, bool is_polygon = true);

    CPT_CPU void setup(const std::array<std::vector<Vec3>, 3>& prims, bool is_polygon = true);

    CPT_CPU_GPU_INLINE int sample_emitter_primitive(uint32_t sample, float& pdf) const {
        pdf *= inv_area;
        return (sample % uint32_t(prim_num)) + prim_offset;
    }

    CPT_GPU_INLINE float solid_angle_pdf(const Vec3& normal, const Vec3& incid_dir, float min_depth) const {
        return inv_area * min_depth * min_depth / max(fabsf(incid_dir.dot(normal)), 1e-4);
    }

    CPT_CPU_GPU_INLINE bool is_emitter() const noexcept { return this->emitter_id > 0; }
    CPT_CPU_GPU ObjInfo(int bsdf_id, int prim_off, int prim_num, uint8_t emitter_id = 0):
        _aabb(1e5, -1e5, -1, -1), bsdf_id(bsdf_id), 
        prim_offset(prim_off), prim_num(prim_num), emitter_id(emitter_id)
    {}

    CPT_CPU void export_bound(Vec3& mini, Vec3& maxi) const noexcept;
};