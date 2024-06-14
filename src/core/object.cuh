/**
 * CUDA object information
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include "core/bsdf.cuh"
#include "core/aos.cuh"
#include "core/shapes.cuh"

class ObjInfo {
private:
    AABB _aabb;
public:
    int bsdf_id;            // index pf the current object
    int prim_offset;        // offset to the start of the primitives
    int prim_num;           // number of primitives
    float inv_area;         // inverse area
    uint8_t emitter_id;     // index to the emitter, 0xff means not an emitter
public:
    CPT_CPU_GPU_INLINE bool intersect(const Ray& ray, float& t_near) const noexcept {
        return _aabb.intersect(ray, t_near);
    }

    CPT_CPU_GPU void setup(const ArrayType<Vec3>& prims, bool is_polygon = true) {
        int ub = prim_offset + prim_num;
        for (int i = prim_offset; i < ub; i++) {
            if (is_polygon) {
                _aabb.mini.minimize(prims.x(i));
                _aabb.mini.minimize(prims.y(i));
                _aabb.mini.minimize(prims.z(i));

                _aabb.maxi.maximize(prims.x(i));
                _aabb.maxi.maximize(prims.y(i));
                _aabb.maxi.maximize(prims.z(i));
                inv_area += (prims.y(i) - prims.x(i)).cross(prims.z(i) - prims.x(i)).length();
            } else {
                _aabb.mini = prims.x(i) - prims.y(i).x();
                _aabb.maxi = prims.x(i) + prims.y(i).x();
                inv_area = static_cast<float>(4.f * M_Pi) * prims.y(i).x() * prims.y(i).x();
            }
        }
        if (is_polygon)
            inv_area *= 0.5f;
        inv_area = 1.f / inv_area;
    }

    CPT_CPU void setup(const std::array<std::vector<Vec3>, 3>& prims, bool is_polygon = true) {
        int ub = prim_offset + prim_num;
        for (int i = prim_offset; i < ub; i++) {
            if (is_polygon) {
                _aabb.mini.minimize(prims[0][i]);
                _aabb.mini.minimize(prims[1][i]);
                _aabb.mini.minimize(prims[2][i]);

                _aabb.maxi.maximize(prims[0][i]);
                _aabb.maxi.maximize(prims[1][i]);
                _aabb.maxi.maximize(prims[2][i]);
                inv_area += (prims[1][i] - prims[0][i]).cross(prims[2][i] - prims[0][i]).length();
            } else {
                _aabb.mini = prims[0][i] - prims[1][i].x();
                _aabb.maxi = prims[0][i] + prims[1][i].x();
                inv_area = static_cast<float>(4.f * M_Pi) * prims[1][i].x() * prims[1][i].x();
            }
        }
        if (is_polygon)
            inv_area *= 0.5f;
        inv_area = 1.f / inv_area;
    }

    CPT_CPU_GPU_INLINE int sample_emitter_primitive(uint32_t sample, float& pdf) const {
        pdf *= inv_area;
        return (sample % uint32_t(prim_num)) + prim_offset;
    }

    CPT_GPU_INLINE float solid_angle_pdf(const Vec3& normal, const Vec3& incid_dir, float min_depth) const {
        return inv_area * min_depth * min_depth / max(fabsf(incid_dir.dot(normal)), 1e-4);
    }

    CPT_CPU_GPU_INLINE bool is_emitter() const noexcept { return this->emitter_id > 0; }
    CPT_CPU_GPU ObjInfo(int bsdf_id, int prim_off, int prim_num, uint8_t emitter_id = 0):
        bsdf_id(bsdf_id), prim_offset(prim_off), prim_num(prim_num), emitter_id(emitter_id)
    {}
};