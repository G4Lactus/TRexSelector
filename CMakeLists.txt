# Root CMakeLists
# =============================
# This file defines project-wide settings and includes each subfolder in src/
# It builds below /build/ a:
# - release/  folder: optimized build without debug symbols
# - debug/    folder: unoptimized build with debug symbols and extra checks

cmake_minimum_required(VERSION 3.18)

# Project definition
project(TRexLearning
    VERSION 0.1
    DESCRIPTION "C++ Prototype for TRex Selector"
    LANGUAGES CXX)

# Set compilers (configurable via environment variables or CMake options)
set(CMAKE_C_COMPILER "${CMAKE_C_COMPILER}" CACHE PATH "C compiler")
set(CMAKE_CXX_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "C++ compiler")

# Build type (default to Release if not specified)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, Debug, etc.)" FORCE)
endif()

# Macro definitions (before add_subdirectory is called)
macro(set_config_output_dir target)
    get_filename_component(_subdir "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debug/${SUBDIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/release/${SUBDIR}"
    )
endmacro()

# Find packages with error handling
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(LAPACK REQUIRED)

# Include directories (target-specific includes are preferred)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    /opt/homebrew/include
    /opt/homebrew/opt/openblas/include
    /opt/homebrew/opt/cereal/include
    /opt/homebrew/opt/libomp/include
    /opt/homebrew/opt/superlu/include
    /opt/homebrew/include/benchmark
)

# Link directories (target-specific links are preferred)
link_directories(
    /opt/homebrew/lib
    /opt/homebrew/opt/superlu/lib
)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Global compile options (use target_compile_options for specific targets)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Add each /src subdirectory explicitly
add_subdirectory(src)

# Print configuration summary
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Source Directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build Directory: ${CMAKE_BINARY_DIR}")
