cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

project(cuSEAL  LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")


add_subdirectory(SIGMA)
add_subdirectory(extern/cnpy)
add_subdirectory(extern/jsoncpp)

set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_PYTHON OFF)
add_subdirectory(extern/faiss)


set(CUSEAL_UTIL_FILES "")
add_subdirectory(util)


add_executable(cuSEAL
        main.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(cuSEAL PRIVATE SIGMA::sigma)
target_link_libraries(cuSEAL PRIVATE cnpy)
#target_include_directories(cuSEAL PRIVATE cnpy)
target_link_libraries(cuSEAL PRIVATE jsoncpp)


add_executable(keygen
        example/keygen.cpp
        keygen.h
        keygen.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(keygen PRIVATE SIGMA::sigma)
target_link_libraries(keygen PRIVATE cnpy)
target_link_libraries(keygen PRIVATE jsoncpp)
target_link_libraries(keygen PRIVATE -lcurand)
target_link_libraries(keygen PRIVATE faiss)


add_executable(encrypt
        example/encrypt.cu
        encrypt.h
        encrypt.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(encrypt PRIVATE SIGMA::sigma)
target_link_libraries(encrypt PRIVATE cnpy)
target_link_libraries(encrypt PRIVATE jsoncpp)
target_link_libraries(encrypt PRIVATE -lcurand)
target_link_libraries(encrypt PRIVATE faiss)


add_executable(calculate
        example/calculate.cu
        calculate.h
        calculate.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(calculate PRIVATE SIGMA::sigma)
target_link_libraries(calculate PRIVATE cnpy)
target_link_libraries(calculate PRIVATE jsoncpp)
target_link_libraries(calculate PRIVATE -lcurand)
target_link_libraries(calculate PRIVATE faiss)


add_executable(decrypt
        example/decrypt.cpp
        decrypt.h
        decrypt.cpp
        ${CUSEAL_UTIL_FILES}
)
target_link_libraries(decrypt PRIVATE SIGMA::sigma)
target_link_libraries(decrypt PRIVATE cnpy)
target_link_libraries(decrypt PRIVATE jsoncpp)
target_link_libraries(decrypt PRIVATE -lcurand)
target_link_libraries(decrypt PRIVATE faiss)
