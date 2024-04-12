cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
# CUDA_ARCHITECTURES native is available in CMake since 3.24

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O0")
ELSE()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
ENDIF()
SET(COMPILE_CUDA True)

if (NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    # required by CLion
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJ_NAME "cuda-pt")

project(${PROJ_NAME} LANGUAGES CUDA CXX)

find_package(Threads REQUIRED)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})
add_definitions(${PNG_DEFINITIONS})
set(LIBRARIES ${LIBRARIES} ${PNG_LIBRARIES})

add_executable(${PROJ_NAME}
        src/basic_pt.cu
        src/ext/lodepng/lodepng.cpp
)

target_include_directories(
        ${PROJ_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(
        ${PROJ_NAME} PRIVATE
        ${PNG_LIBRARY}
)

set_target_properties(
        ${PROJ_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)
# CUDA_SEPARABLE_COMPILATION ON: member function declaration and implementation can be separate
# https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

# CUDA_ARCHITECTURES native: CUDA_ARCHITECTURES is detected automatically
# https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures

target_compile_options(
        ${PROJ_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        >
)