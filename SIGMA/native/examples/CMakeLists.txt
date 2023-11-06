# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.13)

project(SIGMAExamples VERSION 4.1.1 LANGUAGES CXX)

# If not called from root CMakeLists.txt
if(NOT DEFINED SIGMA_BUILD_EXAMPLES)
    set(SIGMA_BUILD_EXAMPLES ON)

    # Import Microsoft SIGMA
    find_package(SIGMA 4.1.1 EXACT REQUIRED)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

if(SIGMA_BUILD_EXAMPLES)
    add_executable(sigmaexamples)

    target_sources(sigmaexamples
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/examples.cpp
            ${CMAKE_CURRENT_LIST_DIR}/1_bfv_basics.cpp
            ${CMAKE_CURRENT_LIST_DIR}/2_encoders.cpp
            ${CMAKE_CURRENT_LIST_DIR}/3_levels.cpp
            ${CMAKE_CURRENT_LIST_DIR}/4_bgv_basics.cpp
            ${CMAKE_CURRENT_LIST_DIR}/5_ckks_basics.cpp
            ${CMAKE_CURRENT_LIST_DIR}/6_rotation.cpp
            ${CMAKE_CURRENT_LIST_DIR}/7_serialization.cpp
            ${CMAKE_CURRENT_LIST_DIR}/8_performance.cpp
    )

    if(TARGET SIGMA::sigma)
        target_link_libraries(sigmaexamples PRIVATE SIGMA::sigma)
    elseif(TARGET SIGMA::sigma_shared)
        target_link_libraries(sigmaexamples PRIVATE SIGMA::sigma_shared)
    else()
        message(FATAL_ERROR "Cannot find target SIGMA::sigma or SIGMA::sigma_shared")
    endif()
endif()
