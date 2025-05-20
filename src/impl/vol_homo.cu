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
 * @brief Homogeneous volume creator
 * @date Unknown
 */
#include "volume/homogeneous.cuh"

CPT_KERNEL void create_homogeneous_volume(Medium **media,
                                          PhaseFunction **phases, int med_id,
                                          int ph_id, Vec4 sigma_a, Vec4 sigma_s,
                                          float scale) {
    if (threadIdx.x == 0) {
        media[med_id] = new HomogeneousMedium(sigma_a * scale, sigma_s * scale);
        media[med_id]->bind_phase_function(phases[ph_id]);
    }
}
