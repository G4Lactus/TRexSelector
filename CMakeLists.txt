# Root CMakeLists.txt
# =============================================================================
# This file defines project-wide settings and includes each subfolder in src/ .
#
# It builds below /build/ a:
# - release/  folder: optimized build without debug symbols
# - debug/    folder: unoptimized build with debug symbols and extra checks
# =============================================================================

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
# Output Directories (UNIFIED approach)
# ============================================================================
# Set output directories at the root level
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)



# ============================================================================
# Project Definition
# ============================================================================
project(TRexLearning
    VERSION 0.1
    DESCRIPTION "C++ Prototype for TRex Selector"
    LANGUAGES CXX
)

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

# Generate compile_commands.json for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



# ============================================================================
# Find packages with priority order
# ============================================================================

# OpenMP (required)
find_package(OpenMP REQUIRED)

# Eigen3 (required)
find_package(Eigen3 REQUIRED NO_MODULE)

# Cereal (required for serialization)
find_package(cereal REQUIRED)
if(NOT cereal_FOUND)
    message(STATUS "Cereal not found via find_package, using manual path")
endif()

# Boost: require at least 1.80
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost 1.80 REQUIRED COMPONENTS headers iostreams)

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
# Eigen version
if(Eigen3_FOUND)
    message(STATUS "  - Eigen3: ${Eigen3_VERSION}")
else()
    message(STATUS "  - Eigen3: NOT FOUND")
endif()
#
# Cereal version
if(cereal_FOUND)
    if(cereal_VERSION)
        message(STATUS "  - Cereal: ${cereal_VERSION}")
    else()
        message(STATUS "  - Cereal: Found (version unknown)")
    endif()
else()
    message(STATUS "  - Cereal: Manual path (/opt/homebrew/opt/cereal)")
endif()
#
# Boost version and components
if(Boost_FOUND)
    message(STATUS "  - Boost: ${Boost_VERSION}")
    message(STATUS "    * Headers: ${Boost_INCLUDE_DIRS}")
    message(STATUS "    * IOStreams: ${Boost_iostreams_FOUND}")
else()
    message(STATUS "  - Boost: NOT FOUND")
endif()
#
# OpenMP version
if(OpenMP_FOUND)
    if(OpenMP_CXX_VERSION)
        message(STATUS "  - OpenMP: ${OpenMP_CXX_VERSION}")
    else()
        message(STATUS "  - OpenMP: Found (version unknown)")
    endif()
    message(STATUS "    * Flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "    * Libraries: ${OpenMP_CXX_LIBRARIES}")
else()
    message(STATUS "  - OpenMP: NOT FOUND")
endif()
#
# BLAS/LAPACK
if(BLAS_FOUND)
    message(STATUS "  - BLAS Vendor: ${BLA_VENDOR}")
    message(STATUS "    * Libraries: ${BLAS_LIBRARIES}")
else()
    message(STATUS "  - BLAS: NOT FOUND")
endif()
#
if(LAPACK_FOUND)
    message(STATUS "  - LAPACK: ${LAPACK_LIBRARIES}")
else()
    message(STATUS "  - LAPACK: NOT FOUND")
endif()
#
message(STATUS "")
message(STATUS "Output Directories:")
message(STATUS "  - Runtime (executables): ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "  - Libraries (static/shared): ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "  - Archives (.a): ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
message(STATUS "==============================================")


# ============================================================================
# Include directories for headers (.hpp files)
# ============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Add Cereal include if not found as package
if(NOT cereal_FOUND)
    include_directories(/opt/homebrew/opt/cereal/include)
endif()



# ============================================================================
# Add subdirectories with build targets (.cpp files)
# ============================================================================
add_subdirectory(src)
