cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
# CUDA_ARCHITECTURES native is available in CMake since 3.24

set(PROJ_NAME "cuda-pt")
project(${PROJ_NAME} LANGUAGES CUDA CXX)

set(CUDA_COMMON_FLAGS --expt-relaxed-constexpr)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"
   OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug"
   OR "${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  set(CMAKE_BUILD_TYPE Debug)
  if(MSVC)
    message(STATUS "MSVC: Building project in DEBUG mode.")
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -maxrregcount=40 -arch=native -g -G")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /g /Od /Zi /MDd")
  else()
    message(STATUS "GCC & G++: Building project in DEBUG mode.")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -Wall -G -O0 -maxrregcount=40 -arch=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O0 -fsanitize=leak")
  endif()
  set(CUDA_CONFIG_FLAGS -G --extended-lambda)
else()
  # to check register usage, please set "-Xptxas -v" to cap the register usage
  # to boost occupancy, set ""
  set(CMAKE_BUILD_TYPE Release)
  if(MSVC)
    if(OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /openmp")
    else()
      message(
        FATAL_ERROR
          "OpenMP can not be found but it is required by wavefront path tracer. Manual overriding is required to skip this exception."
      )
    endif()
    message(STATUS "MSVC: Building project in RELEASE mode.")
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -maxrregcount=40 -lineinfo -arch=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
  else()
    message(STATUS "GCC & G++: Building project in RELEASE mode.")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -O3 --maxrregcount=40 -lineinfo -arch=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    if(OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
      message(
        FATAL_ERROR
          "OpenMP can not be found but it is required by wavefront path tracer. Manual overriding is required to skip this exception."
      )
    endif()
  endif()
  add_compile_definitions(OPENMP_ENABLED)
  set(CUDA_CONFIG_FLAGS --extended-lambda --use_fast_math -Xptxas -O3)
endif()
set(CUDA_FINAL_FLAGS ${CUDA_COMMON_FLAGS} ${CUDA_CONFIG_FLAGS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${CMAKE_CURRENT_LIST_DIR}/src/ext/glfw/include
  ${CMAKE_CURRENT_LIST_DIR}/src/ext/imgui
  ${CMAKE_CURRENT_LIST_DIR}/src/ext/tinyxml2
  ${CMAKE_CURRENT_LIST_DIR}/src/ext/tinyobjloader
  ${CMAKE_CURRENT_LIST_DIR}/src/ext/)

add_subdirectory(src/ext/tinyxml2)
add_subdirectory(src/ext/tinyobjloader)
add_subdirectory(src/ext/imgui)
add_subdirectory(src/ext/glfw)
add_subdirectory(src/ext/nanobind)
add_subdirectory(src/ext/nanovdb)
add_subdirectory(src/ext/atomic_queue)

find_package(OpenGL REQUIRED)
if(MSVC)
  list(APPEND CMAKE_PREFIX_PATH "E:\\glew-2.2.0\\glew")
  set(GLEW_USE_STATIC_LIBS false)
  find_package(GLEW REQUIRED)
else()
  find_package(GLEW REQUIRED)
endif()

include_directories(${GLEW_INCLUDE_DIRS})

aux_source_directory(src/impl CPT_IMPL_SRC)
aux_source_directory(src/pt_impl CPT_PT_IMPL_SRC)
aux_source_directory(src/viewer_impl CPT_VIEWER_IMPL_SRC)

set(RENDER_CORE_SRC ${CPT_IMPL_SRC} ${CPT_PT_IMPL_SRC})

add_library(RenderCore STATIC ${RENDER_CORE_SRC})

set_target_properties(RenderCore PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                            CUDA_ARCHITECTURES native)
target_compile_options(RenderCore
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FINAL_FLAGS}>)

target_link_libraries(RenderCore PRIVATE tinyxml2 tinyobjloader GLEW::glew glfw Threads::Threads
                                         imgui ${OPENGL_gl_LIBRARY} max0x7ba::atomic_queue)

set(EXEC1_NAME "pt")
add_executable(${EXEC1_NAME} app/pt_renderer.cu)

target_link_libraries(${EXEC1_NAME} PRIVATE RenderCore)

set_target_properties(${EXEC1_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                               CUDA_ARCHITECTURES native)

target_compile_options(${EXEC1_NAME}
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FINAL_FLAGS}>)

set(EXEC2_NAME "cpt")
add_executable(${EXEC2_NAME} app/viewer.cu ${CPT_VIEWER_IMPL_SRC})

target_link_libraries(${EXEC2_NAME} PRIVATE RenderCore ${OPENGL_LIBRARIES})

set_target_properties(${EXEC2_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                               CUDA_ARCHITECTURES native)

target_compile_options(${EXEC2_NAME}
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FINAL_FLAGS}>)

option(BUILD_PYTHON_MODULE "Build Python Module" ON)
if(BUILD_PYTHON_MODULE)
  message(STATUS "Building Python Module via Nanobind...")
  # ============================ Python Bindings ============================
  find_package(Python REQUIRED COMPONENTS Development)
  find_package(CUDAToolkit REQUIRED)
  if(Python_FOUND AND CUDAToolkit_FOUND)
    message(STATUS "Python found: Version: ${Python_VERSION}")
    message(STATUS "CUDAToolkit found: Version: ${CUDAToolkit_VERSION}")
    message(
      STATUS
        "Building remote distributed rendering code and nanobind python binds..."
    )
    enable_language(CUDA)

    set(NANOBIND_NDARRAY_SRC_FILES
        src/ext/nanobind/src/nb_func.cpp
        src/ext/nanobind/src/common.cpp
        src/ext/nanobind/src/error.cpp
        src/ext/nanobind/src/nb_internals.cpp
        src/ext/nanobind/src/nb_type.cpp
        src/ext/nanobind/src/nb_static_property.cpp
        src/ext/nanobind/src/implicit.cpp
        src/ext/nanobind/src/nb_ndarray.cpp)

    add_library(
      pyrender MODULE src/pyrender/python_render.cu src/pyrender/python_bind.cu
                      ${NANOBIND_NDARRAY_SRC_FILES})

    target_include_directories(
      pyrender
      PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src
              ${CMAKE_CURRENT_LIST_DIR}/src/ext/nanobind/include
              ${CMAKE_CURRENT_LIST_DIR}/src/ext/nanobind/ext/robin_map/include/
              ${Python_INCLUDE_DIRS})

    target_link_libraries(pyrender PRIVATE ${Python_LIBRARIES} RenderCore)

    if(MSVC)
      set(MODULE_SUFFIX ".pyd")
    else()
      set(MODULE_SUFFIX ".so")
    endif()

    set_target_properties(
      pyrender
      PROPERTIES PREFIX ""
                 SUFFIX ${MODULE_SUFFIX}
                 CUDA_SEPARABLE_COMPILATION ON
                 CUDA_ARCHITECTURES native)

    target_compile_options(
      pyrender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FINAL_FLAGS}>)
  else()
    if(Python_FOUND)
      message(
        WARNING
          "Python Package not found. Without Python Package, remote distributed rendering can not be compiled."
      )
    else()
      message(
        WARNING
          "CUDAToolkit not found. Without CUDAToolkit, remote distributed rendering can not be compiled."
      )
    endif()
  endif()
else()
  message(STATUS "Python Module building disabled.")
endif()

# CUDA_SEPARABLE_COMPILATION ON: member function declaration and implementation
# can be separate
# https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

# CUDA_ARCHITECTURES native: CUDA_ARCHITECTURES is detected automatically
# https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures
