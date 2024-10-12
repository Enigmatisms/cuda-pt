#include "core/object.cuh"

/**
 * CUDA object information
 * @date: 5.20.2024
 * @author: Qianyue He
*/
#pragma once
#include <array>
#include "core/aos.cuh"
#include "core/aabb.cuh"

CPT_CPU_GPU void ObjInfo::setup(const ArrayType<Vec3>& prims, bool is_polygon) {
    int ub = prim_offset + prim_num;
    for (int i = prim_offset; i < ub; i++) {
        if (is_polygon) {
            _aabb.mini.minimized(prims.x(i));
            _aabb.mini.minimized(prims.y(i));
            _aabb.mini.minimized(prims.z(i));

            _aabb.maxi.maximized(prims.x(i));
            _aabb.maxi.maximized(prims.y(i));
            _aabb.maxi.maximized(prims.z(i));
            _aabb.mini -= AABB_EPS;
            _aabb.maxi += AABB_EPS;
            inv_area += (prims.y(i) - prims.x(i)).cross(prims.z(i) - prims.x(i)).length_h();
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

CPT_CPU void ObjInfo::setup(const std::array<std::vector<Vec3>, 3>& prims, bool is_polygon) {
    int ub = prim_offset + prim_num;
    for (int i = prim_offset; i < ub; i++) {
        if (is_polygon) {
            _aabb.mini.minimized(prims[0][i]);
            _aabb.mini.minimized(prims[1][i]);
            _aabb.mini.minimized(prims[2][i]);

            _aabb.maxi.maximized(prims[0][i]);
            _aabb.maxi.maximized(prims[1][i]);
            _aabb.maxi.maximized(prims[2][i]);
            _aabb.mini -= AABB_EPS;
            _aabb.maxi += AABB_EPS;
            inv_area += (prims[1][i] - prims[0][i]).cross(prims[2][i] - prims[0][i]).length_h();
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

CPT_CPU void ObjInfo::export_bound(Vec3& mini, Vec3& maxi) const noexcept {
    mini.minimized(_aabb.mini);
    maxi.maximized(_aabb.maxi);
}
