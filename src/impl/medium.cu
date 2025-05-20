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
 * @brief Medium Info
 * @date 2025-2-17
 */
#include "core/medium.cuh"

const std::array<const char *, NumSupportedMedium> MEDIUM_NAMES = {
    "Homogeneous", "Grid Volume"};

const std::array<const char *, NumSupportedPhase> PHASES_NAMES = {
    "NullForward", "Isotropic", "HenyeyGreenstein",
    "DuoHG",       "Rayleigh",  "SGGX"};
