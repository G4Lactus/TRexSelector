# Root CMakeLists
# =============================
# This file defines project-wide settings and includes each subfolder in src/
# It builds below /build/ a:
# - release/  folder: optimized build without debug symbols
# - debug/    folder: unoptimized build with debug symbols and extra checks

cmake_minimum_required(VERSION 3.18)

# ============================================================================
# Compiler Selection (MUST be before project())
# ============================================================================
# On macOS, prefer Homebrew LLVM over Apple Clang
if(APPLE AND NOT DEFINED CMAKE_CXX_COMPILER)
    set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")

    if(EXISTS "${HOMEBREW_LLVM_PREFIX}/bin/clang++")
        set(CMAKE_C_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang" CACHE FILEPATH "C compiler")
        set(CMAKE_CXX_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "C++ compiler")
        message(STATUS "*** Using Homebrew LLVM: ${CMAKE_CXX_COMPILER}")
    else()
        message(WARNING "*** Homebrew LLVM not found at ${HOMEBREW_LLVM_PREFIX}")
        message(WARNING "*** Install with: brew install llvm")
        message(WARNING "*** Falling back to system compiler")
    endif()
endif()



# ============================================================================
# Project Definition
# ============================================================================
project(TRexLearning
    VERSION 0.1
    DESCRIPTION "C++ Prototype for TRex Selector"
    LANGUAGES CXX)

# Build type (default to Release if not specified)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, Debug, etc.)" FORCE)
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")


# ============================================================================
# Find packages with priority order
# ============================================================================

# OpenMP (required)
find_package(OpenMP REQUIRED)

# Eigen3 (required)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Cereal (required for serialization)
find_package(cereal REQUIRED)
if(NOT cereal_FOUND)
    message(STATUS "Cereal not found via find_package, using manual path")
endif()

# BLAS/LAPACK with priority order
# Priority: Accelerate > OpenBLAS > Reference BLAS
set(BLA_VENDOR Apple)  # Try Apple Accelerate first (fastest on macOS)
find_package(BLAS)
find_package(LAPACK)

if(NOT BLAS_FOUND)
    # Fallback to OpenBLAS
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "==============================================")
message(STATUS "Build Configuration")
message(STATUS "==============================================")
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Compiler:")
message(STATUS "  - ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  - Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  - Path: ${CMAKE_CXX_COMPILER}")
message(STATUS "")
message(STATUS "Libraries:")
message(STATUS "  - BLAS Vendor: ${BLA_VENDOR}")
message(STATUS "  - BLAS: ${BLAS_LIBRARIES}")
message(STATUS "  - OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "==============================================")


# ============================================================================
# Include directories (only for headers without CMake packages)
# ============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Add Cereal include if not found as package
if(NOT cereal_FOUND)
    include_directories(/opt/homebrew/opt/cereal/include)
endif()

# Add subdirectories
add_subdirectory(src)
