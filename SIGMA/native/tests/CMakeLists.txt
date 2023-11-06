# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.13)

project(SIGMATest VERSION 4.1.1 LANGUAGES CXX C)

# If not called from root CMakeLists.txt
if(NOT DEFINED SIGMA_BUILD_TESTS)
    set(SIGMA_BUILD_TESTS ON)

    # Import Microsoft SIGMA
    find_package(SIGMA 4.1.1 EXACT REQUIRED)

    # Must define these variables and include macros
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${OUTLIB_PATH})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(SIGMA_THIRDPARTY_DIR ${CMAKE_CURRENT_LIST_DIR}/../../thirdparty)
    set(THIRDPARTY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty)
    include(FetchContent)
    mark_as_advanced(FETCHCONTENT_BASE_DIR)
    mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
    mark_as_advanced(FETCHCONTENT_QUIET)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../cmake)
    include(SIGMAMacros)
else()
    set(THIRDPARTY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../thirdparty)
endif()

if(NOT DEFINED SIGMA_BUILD_DEPS)
    # [option] SIGMA_BUILD_DEPS (default: ON)
    # Download and build missing dependencies, throw error if disabled.
    set(SIGMA_BUILD_DEPS_OPTION_STR "Automatically download and build unmet dependencies")
    option(SIGMA_BUILD_DEPS ${SIGMA_BUILD_DEPS_OPTION_STR} ON)
endif()

# if SIGMA_BUILD_TESTS is ON, use GoogleTest
if(SIGMA_BUILD_TESTS)
    if(SIGMA_BUILD_DEPS)
        sigma_fetch_thirdparty_content(ExternalGTest)
        add_library(GTest::gtest ALIAS gtest)
    else()
        find_package(GTest 1 CONFIG)
        if(NOT GTest_FOUND)
            message(FATAL_ERROR "GoogleTest: not found")
        else()
            message(STATUS "GoogleTest: found")
        endif()
    endif()

    add_executable(sigmatest "")

    add_subdirectory(sigma)

    if(TARGET SIGMA::sigma)
        target_link_libraries(sigmatest PRIVATE SIGMA::sigma GTest::gtest)
    elseif(TARGET SIGMA::sigma_shared)
        target_link_libraries(sigmatest PRIVATE SIGMA::sigma_shared GTest::gtest)
    else()
        message(FATAL_ERROR "Cannot find target SIGMA::sigma or SIGMA::sigma_shared")
    endif()

    # In Debug mode, enable AddressSanitizer (and LeakSanitizer) on Unix-like platforms.
    if(SIGMA_DEBUG AND UNIX)
        # On macOS, only AddressSanitizer is enabled.
        # On Linux, LeakSanitizer is enabled by default.
        target_compile_options(sigmatest PUBLIC -fsanitize=address)
        target_link_options(sigmatest PUBLIC -fsanitize=address)
        if(NOT APPLE)
            message(STATUS "Sanitizers enabled: address, leak")
        else()
            message(STATUS "Sanitizers enabled: address")
        endif()
    endif()
endif()
