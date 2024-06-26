cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
# CUDA_ARCHITECTURES native is available in CMake since 3.24

set(PROJ_NAME "cuda-pt")
project(${PROJ_NAME} LANGUAGES CUDA CXX)

option(USE_SOA "Use struct of arrays" OFF)
if (USE_SOA)
  add_definitions(-DUSE_SOA)
  message(STATUS "Using SoA3 (struct of arrays) for GPU containers.")
else()
  message(STATUS "Using AoS3 (array of structs) for GPU containers.")
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -g -G -O0")
  set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O0")
  message(STATUS "Building project in DEBUG mode.")
  set(CMAKE_BUILD_TYPE Debug)
ELSE()
  # to check register usage, please set "-Xptxas -v"
  # to cap the register usage to boost occupancy, set ""
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --maxrregcount=64 -lineinfo")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
  message(STATUS "Building project in RELEASE mode.")
  set(CMAKE_BUILD_TYPE Release)
ENDIF()
SET(COMPILE_CUDA True)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src ${CMAKE_CURRENT_LIST_DIR}/src/ext/variant)

add_subdirectory(src/ext/tinyxml2)
add_subdirectory(src/ext/tinyobjloader)

aux_source_directory(
        ${CMAKE_CURRENT_LIST_DIR}/src/impl CUPT_IMPL
)

set(EXEC1_NAME "depth")
add_executable(${EXEC1_NAME}
        app/depth_renderer.cu
        src/ext/lodepng/lodepng.cpp
        ${CUPT_IMPL}
)

target_link_libraries(
        ${EXEC1_NAME} PRIVATE
        tinyxml2 tinyobjloader
)

set_target_properties(
        ${EXEC1_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)

target_compile_options(
        ${EXEC1_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        >
)

set(EXEC2_NAME "pt")
add_executable(${EXEC2_NAME}
        app/pt_renderer.cu
        src/ext/lodepng/lodepng.cpp
        ${CUPT_IMPL}
)

target_link_libraries(
        ${EXEC2_NAME} PRIVATE
        tinyxml2 tinyobjloader
)

set_target_properties(
        ${EXEC2_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)

target_compile_options(
        ${EXEC2_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        >
)

# ========================================= unit-test ===============================================
set(TEST1_NAME "test_so3")
add_executable(${TEST1_NAME}
        test/test_so3.cu
)

set(TEST2_NAME "variant_test")
add_executable(${TEST2_NAME}
        test/variant_test.cu
)

set_target_properties(
        ${TEST1_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)

add_executable(pt_test
        test/path_tracer_test.cu
        src/ext/lodepng/lodepng.cpp
        ${CUPT_IMPL}
)

target_link_libraries(
        pt_test PRIVATE
        tinyxml2 tinyobjloader
)

set_target_properties(
        pt_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
)

target_compile_options(
        pt_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        >
)

add_executable(sensor_test test/parsers/sensor_parse.cc)

target_link_libraries(sensor_test PRIVATE tinyxml2)

add_executable(obj_test test/parsers/obj_parser.cu)

target_link_libraries(obj_test PRIVATE tinyxml2 tinyobjloader)


# CUDA_SEPARABLE_COMPILATION ON: member function declaration and implementation can be separate
# https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

# CUDA_ARCHITECTURES native: CUDA_ARCHITECTURES is detected automatically
# https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures