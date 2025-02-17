#!/bin/bash

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