cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(cuSEAL  LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(cuSEAL main.cpp)

add_subdirectory(SIGMA)
target_link_libraries(cuSEAL PRIVATE SIGMA::sigma)

#find_package(CUDA REQUIRED)
#target_link_libraries(sigma PRIVATE cufft)
