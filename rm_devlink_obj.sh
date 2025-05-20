#!/bin/bash

# Copyright (C) 2025 Qianyue He
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General
# Public License along with this program. If not, see
#
#             <https://www.gnu.org/licenses/>.

BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "'$BUILD_DIR' folder not found. Creating it..."
    mkdir "$BUILD_DIR"
else
    # remove all the device-link.obj file, since if the kernel updates
    # *.device-link.obj won't update, which might cause problems
    # of 'cuda fatbin' linking error
    find "$BUILD_DIR" -type f -name "*.device-link.obj" -exec rm -v {} +
fi

# cd "$BUILD_DIR"
# cmake -DCMAKE_BUILD_TYPE=release ..
# cmake --build . --config Release --parallel 15 -j 15
