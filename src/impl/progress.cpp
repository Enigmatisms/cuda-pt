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
 * @author Unknown
 * @brief Solution from
 * https://stackoverflow.com/questions/14539867/how-to-display-a-progress-indicator-in-pure-c-c-cout-printf
 * C++ progress bar
 * @date Unknown
 */
#include "core/progress.h"

#define PBSTR                                                                  \
    "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" \
    "||"                                                                       \
    "||||||||||||||||||||||||||"
#define PBWIDTH 100

void printProgress(int spp_now, int spp_full) {
    float percentage = float(spp_now + 1) / float(spp_full);
    int val = (int)(percentage * 100);
    int lpad = (int)(percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% (%4d / %4d) [%.*s%*s]", val, spp_now + 1, spp_full, lpad,
           PBSTR, rpad, "");
    fflush(stdout);
}
