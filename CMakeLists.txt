cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(cuSEAL  LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")


add_subdirectory(SIGMA)
add_subdirectory(extern/cnpy)
add_subdirectory(extern/jsoncpp)


set(CUSEAL_UTIL_FILES "")
add_subdirectory(util)


add_executable(cuSEAL
        main.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(cuSEAL PRIVATE SIGMA::sigma)
target_link_libraries(cuSEAL PRIVATE cnpy)
target_include_directories(cuSEAL PRIVATE cnpy)
target_link_libraries(cuSEAL PRIVATE jsoncpp)


add_executable(keygen
        keygen.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(keygen PRIVATE SIGMA::sigma)
target_link_libraries(keygen PRIVATE cnpy)
target_link_libraries(keygen PRIVATE jsoncpp)


add_executable(encrypt
        encrypt.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(encrypt PRIVATE SIGMA::sigma)
target_link_libraries(encrypt PRIVATE cnpy)
target_link_libraries(encrypt PRIVATE jsoncpp)


add_executable(calculateIP
        calculateIP.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(calculateIP PRIVATE SIGMA::sigma)
target_link_libraries(calculateIP PRIVATE cnpy)
target_link_libraries(calculateIP PRIVATE jsoncpp)

