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

/**
 * @author Qianyue He
 * @brief Includes all the derived class of Medium
 * @date 2025.02.09
 */

#pragma once

#include "core/enums.cuh"
#include "volume/grid.cuh"
#include "volume/homogeneous.cuh"

template <typename MedType, typename... Args>
CPT_KERNEL void create_device_medium(Medium **dst, PhaseFunction **phases,
                                     int med_id, int ph_id, Args... args) {
    static_assert(std::is_base_of_v<Medium, MedType>,
                  "MedType must be derived from Medium");

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[med_id] = new MedType(args...);
        dst[med_id]->bind_phase_function(phases[ph_id]);
    }
}

CPT_KERNEL void load_homogeneous_kernel(Medium **dst, Vec4 sigma_a,
                                        Vec4 sigma_s, float scale);

CPT_KERNEL void load_grid_kernel(Medium **dst, Vec4 const_alb, float scale,
                                 float tp_scale, float em_scale);

CPT_KERNEL void bind_phase_func_kernel(Medium **dst, PhaseFunction **phases,
                                       int ph_id);
