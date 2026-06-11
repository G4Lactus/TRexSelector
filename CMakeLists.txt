# =============================================================================
# Project Root CMakeLists.txt
# =============================================================================
# This file defines project-wide settings and includes each subfolder in src/ .
#
# It builds below /build/ a:
# - release/  folder: optimized build without debug symbols
# - debug/    folder: unoptimized build with debug symbols and extra checks
# =============================================================================
cmake_minimum_required(VERSION 3.23...4.3)


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
# Output Directories
# ============================================================================
# Set output directories at the root level
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)



# ============================================================================
# Project Definition
# ============================================================================
project(TRexSelector
    VERSION 0.1.0
    DESCRIPTION "C++ Implementation of the T-Rex Selector"
    LANGUAGES CXX
)

# Build type (default to Release if not specified)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, Debug, etc.)" FORCE)
endif()

# ============================================================================
# User-configurable options
# ============================================================================
option(TREX_USE_BLAS_LAPACK "Use BLAS/LAPACK backends for Eigen" ON)
option(TREX_PORTABLE_BUILD "Disable -march=native for portable/wheel builds. Set ON when building for PyPI or CI." OFF)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Generate compile_commands.json for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Python / IDE Include Paths (For VS Code IntelliSense)
# ============================================================================
# These are explicitly added to the IDE's compile_commands.json so that
# VS Code stops reporting pybind11 errors in Python/src/.
include_directories("/opt/homebrew/Caskroom/miniconda/base/envs/trex-dev/include/python3.12")
include_directories("/opt/homebrew/Caskroom/miniconda/base/envs/trex-dev/lib/python3.12/site-packages/pybind11/include")


# ============================================================================
# Find packages with priority order
# ============================================================================

# OpenMP (required)
# Pin libomp to the stable Homebrew symlink — survives future `brew upgrade llvm`
if(APPLE)
    if(NOT DEFINED HOMEBREW_LLVM_PREFIX)
        set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")
    endif()
    set(OpenMP_CXX_FLAGS       "-fopenmp=libomp -I${HOMEBREW_LLVM_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES   "libomp")
    set(OpenMP_libomp_LIBRARY  "${HOMEBREW_LLVM_PREFIX}/lib/libomp.dylib"
        CACHE FILEPATH "libomp path - pinned to stable Homebrew symlink" FORCE)
endif()
find_package(OpenMP REQUIRED)

# Eigen3 (required)
find_package(Eigen3 REQUIRED NO_MODULE)

# Cereal (required for serialization)
find_package(cereal REQUIRED)

# Boost: require at least 1.80
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost 1.80 REQUIRED COMPONENTS headers)

if(TREX_USE_BLAS_LAPACK)
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
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Add Cereal include if not found as package
if(NOT cereal_FOUND)
    include_directories(/opt/homebrew/opt/cereal/include)
endif()


# ============================================================================
# Add subdirectories with build targets (.cpp files)
# ============================================================================
include(CTest)
enable_testing()
option(TREX_BUILD_TESTS "Build C++ test executables" ON)
add_subdirectory(src)
if (TREX_BUILD_TESTS)
    add_subdirectory(tests)
endif()


# ============================================================================
# Package Configuration  (enables find_package(TRexSelector))
# ============================================================================
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)   # gives portable CMAKE_INSTALL_LIBDIR / INCLUDEDIR

# 1. Generate TRexSelectorConfig.cmake from the .in template
configure_package_config_file(
    cmake/TRexSelectorConfig.cmake.in                           # source template
    ${CMAKE_CURRENT_BINARY_DIR}/TRexSelectorConfig.cmake        # output in build/
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TRexSelector
)

# 2. Generate TRexSelectorConfigVersion.cmake
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TRexSelectorConfigVersion.cmake
    VERSION         ${PROJECT_VERSION}
    COMPATIBILITY   SameMajorVersion
)

# 3. Install both config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TRexSelectorConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TRexSelectorConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TRexSelector
)

# 4. Install the exported target manifest
install(EXPORT TRexSelectorTargets
    FILE        TRexSelectorTargets.cmake
    NAMESPACE   TRexSelector::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TRexSelector
)


# ============================================================================
# CPack (source + binary packaging)
# ============================================================================
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CPACK_PACKAGE_NAME                 "TRexSelector")
    set(CPACK_PACKAGE_VENDOR               "Robust Data Science Group - TU Darmstadt")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY  "C++ T-Rex Selector for FDR-controlled variable selection in high-dimensional regression")
    set(CPACK_PACKAGE_VERSION              ${PROJECT_VERSION})
    set(CPACK_PACKAGE_VERSION_MAJOR        ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR        ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH        ${PROJECT_VERSION_PATCH})
    set(CPACK_RESOURCE_FILE_LICENSE        "${PROJECT_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_README         "${PROJECT_SOURCE_DIR}/README.md")

    # Binary package  (cmake --build <dir> --target package)
    # Produces TRexSelector-<version>-<system>.tar.gz and .zip
    # Contains: compiled libs, headers, CMake package config files
    set(CPACK_GENERATOR "TGZ;ZIP")

    # Source package  (cmake --build <dir> --target package_source)
    # Contains: src/, cmake/, CMakeLists.txt, CMakePresets.json, LICENSE, README.md
    # Excludes: build artefacts, simulation outputs, R/Python tooling, notes
    set(CPACK_SOURCE_GENERATOR          "TGZ;ZIP")
    set(CPACK_SOURCE_PACKAGE_FILE_NAME  "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-src")
    set(CPACK_SOURCE_IGNORE_FILES
        "/\\.git/"
        "/build/"
        "/install/"
        "/simulations/"
        "/R/"
        "/Python/"
        "/notes/"
        "/tests/"
        "\\.DS_Store"
    )

    include(CPack)
endif()


# ============================================================================
# VS Code / IDE IntelliSense Dummy Target (DO NOT BUILD)
# ============================================================================
# This simply forces ms-vscode.cmake-tools to parse and provide IntelliSense for
# the R package source files that are otherwise built externally by `R CMD INSTALL`.
file(GLOB_RECURSE R_BINDINGS_SOURCES "R/TRexSelector/src/*.cpp")
add_library(r_bindings_ide_only EXCLUDE_FROM_ALL ${R_BINDINGS_SOURCES})
target_include_directories(r_bindings_ide_only PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "/Library/Frameworks/R.framework/Resources/include"
    "/Library/Frameworks/R.framework/Resources/library/Rcpp/include"
    "/Library/Frameworks/R.framework/Resources/library/RcppEigen/include"
    "/Library/Frameworks/R.framework/Resources/library/BH/include"
    "/Library/Frameworks/R.framework/Resources/library/Rcereal/include"
)
