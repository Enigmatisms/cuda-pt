cmake_minimum_required(VERSION 3.16)

# 项目名称
project(OptiXExample LANGUAGES CUDA CXX)

# CUDA 和 OptiX 配置
find_package(CUDAToolkit 12.1 REQUIRED)
# find_package(OptiX REQUIRED)

# 设置 OptiX 头文件路径
set(OPTIX_INCLUDE_DIRS "E:/optix/include")
# 设置 MSVC 编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc true")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# 设置包含目录
include_directories(${OPTIX_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

# 编译源文件
add_executable(optix_example main.cpp)

# 添加 CUDA 和 OptiX 链接
target_link_libraries(optix_example
    PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)