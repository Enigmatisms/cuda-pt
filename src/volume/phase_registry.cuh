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
 * @brief Includes all the derived class of PhaseFunction
 * @date 2025.02.09
 */
#pragma once

#include "core/enums.cuh"
#include "volume/henyey_greenstein.cuh"
#include "volume/rayleigh.cuh"
#include "volume/sggx.cuh"

template <typename PhaseType, typename... Args>
CPT_KERNEL void create_device_phase(PhaseFunction **dst, int index,
                                    Args... args) {
    static_assert(std::is_base_of_v<PhaseFunction, PhaseType>,
                  "PhaseType must be derived from PhaseFunction");

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (dst[index])
            delete dst[index];
        dst[index] = new PhaseType(args...);
    }
}

CPT_KERNEL void load_phase_kernel(PhaseFunction **dst, int index, Vec4 data);
