# =============================================================================
# Project Root CMakeLists.txt
# =============================================================================
# This file defines project-wide settings and includes each subfolder in src/ .
#
# It builds below /build/ a:
# - release/  folder: optimized build without debug symbols
# - debug/    folder: unoptimized build with debug symbols and extra checks
# =============================================================================
cmake_minimum_required(VERSION 3.26...4.3)


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
        message(WARNING "*** Falling back to system compiler (single-threaded if it has no OpenMP)")
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
    VERSION 1.0.0
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
# On Windows there is no system BLAS/LAPACK to find (Accelerate/OpenBLAS), so
# default to Eigen's built-in backends there.
if(WIN32)
    option(TREX_USE_BLAS_LAPACK "Use BLAS/LAPACK backends for Eigen" OFF)
else()
    option(TREX_USE_BLAS_LAPACK "Use BLAS/LAPACK backends for Eigen" ON)
endif()
option(TREX_PORTABLE_BUILD "Disable -march=native for portable/wheel builds. Set ON when building for PyPI or CI." OFF)
# The SDK install (headers, static libraries, CMake package files) is OFF for Python wheels,
# which need only the extension module. TREX_WITH_OPENMP=OFF forces the single-threaded build.
option(TREX_INSTALL_CPP_SDK "Install the C++ SDK (headers, static libraries, CMake package files)" ON)
option(TREX_WITH_OPENMP "Use OpenMP when it is found (OFF forces the single-threaded build)" ON)

# Shared per-target compiler flags (GCC/Clang vs MSVC)
include(cmake/TRexCompilerOptions.cmake)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# No C++20 modules used; skip MSBuild's per-TU module dependency scan
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
# Generate compile_commands.json for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Find packages with priority order
# ============================================================================

# OpenMP (optional)
# Without OpenMP the library builds single-threaded: the pragmas are ignored
# and utils/openmp/utils_openmp.hpp stubs the omp_* calls (one thread).
# TREX_WITH_OPENMP=OFF forces that path even where OpenMP exists.
# TREX_OPENMP_ENABLED records the outcome for the targets and the package config.
set(TREX_OPENMP_ENABLED OFF)
if(TREX_WITH_OPENMP)
    # macOS: pin libomp to the stable Homebrew LLVM symlink (survives
    # `brew upgrade llvm`), only for Homebrew's clang and when that libomp exists
    if(APPLE)
        if(NOT DEFINED HOMEBREW_LLVM_PREFIX)
            set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")
        endif()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
           EXISTS "${HOMEBREW_LLVM_PREFIX}/lib/libomp.dylib")
            set(OpenMP_CXX_FLAGS       "-fopenmp=libomp -I${HOMEBREW_LLVM_PREFIX}/include")
            set(OpenMP_CXX_LIB_NAMES   "libomp")
            set(OpenMP_libomp_LIBRARY  "${HOMEBREW_LLVM_PREFIX}/lib/libomp.dylib"
                CACHE FILEPATH "libomp path - pinned to stable Homebrew symlink" FORCE)
        endif()
    endif()
    if(MSVC)
        if(CMAKE_VERSION VERSION_LESS 3.30)
            message(FATAL_ERROR "MSVC builds need CMake >= 3.30 (OpenMP_RUNTIME_MSVC support)")
        endif()
        # MSVC's default OpenMP 2.0 rejects the unsigned loop indices used in the
        # solvers (error C3016); the LLVM runtime (-openmp:llvm) supports them.
        set(OpenMP_RUNTIME_MSVC "llvm")
    endif()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(TREX_OPENMP_ENABLED ON)
    else()
        message(STATUS "OpenMP not found: building single-threaded")
    endif()
else()
    message(STATUS "OpenMP disabled (TREX_WITH_OPENMP=OFF): building single-threaded")
endif()
# Single-threaded build: the inactive `#pragma omp` lines would warn once each
if(NOT TREX_OPENMP_ENABLED)
    if(MSVC)
        add_compile_options(/wd4068)
    else()
        add_compile_options(-Wno-unknown-pragmas)
    endif()
endif()

# Eigen3 (required)
find_package(Eigen3 REQUIRED NO_MODULE)

# Cereal (required for serialization)
find_package(cereal REQUIRED)

# Boost: require at least 1.80.
# Use the legacy FindBoost module (module mode) rather than config mode: it
# locates header-only Boost installs that ship no BoostConfig.cmake (e.g. the
# manylinux Python-wheel builds that drop Boost headers into /usr/local/include),
# while still finding Homebrew/apt Boost. CMake 3.30+ defaults to config mode
# (CMP0167 NEW), which fails on such header-only installs.
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()
# No COMPONENTS: Boost is used header-only (Boost.Interprocess / Boost.Random /
# Boost.Math). The FindBoost module still provides the Boost::headers target
# (CMake >= 3.15). Requesting "COMPONENTS headers" would fail in module mode,
# since header-only Boost has no designated component library.
find_package(Boost 1.80 REQUIRED)

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
if(TREX_OPENMP_ENABLED)
    if(OpenMP_CXX_VERSION)
        message(STATUS "  - OpenMP: ${OpenMP_CXX_VERSION}")
    else()
        message(STATUS "  - OpenMP: Found (version unknown)")
    endif()
    message(STATUS "    * Flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "    * Libraries: ${OpenMP_CXX_LIBRARIES}")
elseif(TREX_WITH_OPENMP)
    message(STATUS "  - OpenMP: NOT FOUND (single-threaded build)")
else()
    message(STATUS "  - OpenMP: OFF (TREX_WITH_OPENMP=OFF, single-threaded build)")
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
# Install layout and hygiene
# ============================================================================
# Headers install under a NAMESPACED root, <prefix>/include/TRexSelector/, so
# the package never owns generic names such as include/utils inside a shared
# prefix (a conda environment, /usr/local). Consumers receive the directory
# through the exported targets (find_package); manual -I users append
# /TRexSelector.
#
# `cmake --install` only copies files; it never removes headers that were
# renamed or deleted since the previous install, and a stale header (one
# that #includes a since-renamed file) poisons every consumer of the install
# tree. The namespaced root is therefore pruned before the FILE_SET header
# installs — but only when a stamp file proves the tree was written by this
# project: a foreign directory of the same name is refused, never deleted.
# Declared before add_subdirectory(src) so it runs ahead of the header
# installs (CMP0082: install rules execute in declaration order). Skipped
# entirely without the SDK (Python wheels install only the extension).
include(GNUInstallDirs)   # gives portable CMAKE_INSTALL_LIBDIR / INCLUDEDIR
set(TREX_INSTALL_INCLUDE_ROOT "${CMAKE_INSTALL_INCLUDEDIR}/TRexSelector")
if(TREX_INSTALL_CPP_SDK)
    set(_trex_install_stamp "${CMAKE_CURRENT_BINARY_DIR}/TRexSelector.install-stamp")
    file(WRITE "${_trex_install_stamp}"
         "TRexSelector ${PROJECT_VERSION} header tree; pruned and rewritten by cmake --install.\n")
    install(CODE "set(_trex_include_root \"${TREX_INSTALL_INCLUDE_ROOT}\")")
    install(CODE [[
        set(_trex_dir "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${_trex_include_root}")
        if(EXISTS "${_trex_dir}")
            if(NOT EXISTS "${_trex_dir}/TRexSelector.install-stamp")
                message(FATAL_ERROR
                    "TRexSelector: refusing to prune ${_trex_dir}: it carries no "
                    "TRexSelector.install-stamp, so it was not written by this "
                    "project. Remove or rename it by hand if it is stale.")
            endif()
            # Content of the tree about to be pruned, for the timestamp step
            # after the header installs (see below).
            file(GLOB_RECURSE _trex_prev_files RELATIVE "${_trex_dir}" "${_trex_dir}/*")
            foreach(_trex_rel IN LISTS _trex_prev_files)
                string(MAKE_C_IDENTIFIER "${_trex_rel}" _trex_key)
                file(SHA256 "${_trex_dir}/${_trex_rel}" "_trex_prev_sha_${_trex_key}")
            endforeach()
            file(REMOVE_RECURSE "${_trex_dir}")
            message(STATUS "TRexSelector: pruned the package header tree ${_trex_dir}")
        endif()
    ]])
    install(FILES "${_trex_install_stamp}" DESTINATION "${TREX_INSTALL_INCLUDE_ROOT}")
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

# 3. + 4. Install the config files and the exported target manifest — SDK only
#         (the export set is empty without the library install rules).
if(TREX_INSTALL_CPP_SDK)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/TRexSelectorConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/TRexSelectorConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TRexSelector
    )
    install(EXPORT TRexSelectorTargets
        FILE        TRexSelectorTargets.cmake
        NAMESPACE   TRexSelector::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TRexSelector
    )

    # Timestamps of the rewritten header tree. file(INSTALL) gives every
    # installed file the modification time of its source, so a header edited
    # before a consumer object was compiled (against the previous install)
    # looks older than that object after the install, the consumer is not
    # rebuilt, and a changed class layout corrupts its heap. Headers whose
    # content changed (or that are new) since the previous install therefore
    # get the install time; unchanged headers keep theirs, so consumers
    # rebuild exactly what depends on a change. Runs after the header
    # installs of src/ (CMP0082).
    install(CODE [[
        set(_trex_dir "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${_trex_include_root}")
        if(EXISTS "${_trex_dir}")
            file(GLOB_RECURSE _trex_new_files RELATIVE "${_trex_dir}" "${_trex_dir}/*")
            set(_trex_changed 0)
            foreach(_trex_rel IN LISTS _trex_new_files)
                string(MAKE_C_IDENTIFIER "${_trex_rel}" _trex_key)
                file(SHA256 "${_trex_dir}/${_trex_rel}" _trex_sha)
                if(NOT DEFINED "_trex_prev_sha_${_trex_key}" OR
                   NOT _trex_sha STREQUAL "${_trex_prev_sha_${_trex_key}}")
                    file(TOUCH_NOCREATE "${_trex_dir}/${_trex_rel}")
                    math(EXPR _trex_changed "${_trex_changed} + 1")
                endif()
            endforeach()
            message(STATUS "TRexSelector: ${_trex_changed} new or changed file(s) "
                           "in the header tree carry the install time")
        endif()
    ]])
endif()


# ============================================================================
# CPack (source + binary packaging)
# ============================================================================
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CPACK_PACKAGE_NAME                 "TRexSelector")
    set(CPACK_PACKAGE_VENDOR               "Robust Data Science Group - TU Darmstadt")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
        "C++ T-Rex Selector for FDR-controlled variable selection in high-dimensional regression")
    set(CPACK_PACKAGE_VERSION              ${PROJECT_VERSION})
    set(CPACK_PACKAGE_VERSION_MAJOR        ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR        ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH        ${PROJECT_VERSION_PATCH})
    set(CPACK_RESOURCE_FILE_LICENSE        "${PROJECT_SOURCE_DIR}/../LICENSE")
    set(CPACK_RESOURCE_FILE_README         "${PROJECT_SOURCE_DIR}/../README.md")

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
file(GLOB_RECURSE R_BINDINGS_SOURCES "../R/TRexSelector/src/*.cpp")
if(R_BINDINGS_SOURCES)
add_library(r_bindings_ide_only EXCLUDE_FROM_ALL ${R_BINDINGS_SOURCES})
target_include_directories(r_bindings_ide_only PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "/Library/Frameworks/R.framework/Resources/include"
    "/Library/Frameworks/R.framework/Resources/library/Rcpp/include"
    "/Library/Frameworks/R.framework/Resources/library/RcppEigen/include"
    "/Library/Frameworks/R.framework/Resources/library/BH/include"
    "/Library/Frameworks/R.framework/Resources/library/Rcereal/include"
)
endif()
