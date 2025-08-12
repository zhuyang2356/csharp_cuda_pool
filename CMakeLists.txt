cmake_minimum_required(VERSION 3.18)
# CUDA路径（可自定义）
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR "D:/progfiles/cudatools")
endif()
set(CMAKE_CUDA_ARCHITECTURES "70;75;80")  # Volta + Turing + Ampere
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
project(CudaMedianFilter LANGUAGES CXX CUDA)
# 设置输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/inc)
# 查找所有源文件
file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cu ${CMAKE_SOURCE_DIR}/src/*.cpp)
# 生成DLL
add_library(CudaMedianFilter SHARED ${SRC_FILES})
# 设置C++标准
set_target_properties(CudaMedianFilter PROPERTIES
    CXX_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
# 导出函数以便C#调用
if (WIN32)
    set_target_properties(CudaMedianFilter PROPERTIES PREFIX "" SUFFIX ".dll")
endif()