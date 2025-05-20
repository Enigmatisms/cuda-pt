// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

#include "core/object.cuh"

/**
 * @author: Qianyue He
 * @brief CUDA object information
 * @date: 5.20.2024
 */
#include "core/aabb.cuh"
#include "core/aos.cuh"
#include <array>

CPT_CPU void ObjInfo::setup(const std::array<std::vector<Vec3>, 3> &prims,
                            bool is_polygon) {
    int ub = prim_offset + prim_num;
    inv_area = 0;
    for (int i = prim_offset; i < ub; i++) {
        if (is_polygon) {
            _aabb.mini.minimized(prims[0][i]);
            _aabb.mini.minimized(prims[1][i]);
            _aabb.mini.minimized(prims[2][i]);

            _aabb.maxi.maximized(prims[0][i]);
            _aabb.maxi.maximized(prims[1][i]);
            _aabb.maxi.maximized(prims[2][i]);
            inv_area += (prims[1][i] - prims[0][i])
                            .cross(prims[2][i] - prims[0][i])
                            .length_h();
        } else {
            _aabb.mini = prims[0][i] - prims[1][i].x();
            _aabb.maxi = prims[0][i] + prims[1][i].x();
            inv_area = static_cast<float>(4.f * M_Pi) * prims[1][i].x() *
                       prims[1][i].x();
        }
    }
    if (is_polygon) {
        _aabb.mini -= AABB_EPS;
        _aabb.maxi += AABB_EPS;
        inv_area *= 0.5f;
    }
    inv_area = 1.f / inv_area;
}
