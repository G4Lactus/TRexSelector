# src/utils/CMakeLists.txt
# ==================================================================================
# CMake configuration for the utils module.
#
# This library provides foundational utilities for the T-Rex framework.
#
# Organization:
# -------------
# src/utils/
# |-- datagen/
# |-- eval_metrics/
# |-- memmap/
# |-- openmp/
# |-- serialization/
# ==================================================================================


# ==================================================================================
# 1. Library Target
# ==================================================================================
# Create a unified static library for foundational utilities
add_library(trex_utils STATIC "")

# Add alias for consistent naming (mirrors C++ namespace)
add_library(trex::utils ALIAS trex_utils)
# ==================================================================================


# ==================================================================================
# 2. Include Directories
# ==================================================================================
target_include_directories(trex_utils PUBLIC
    # Build-tree
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>

    # Install-tree
    $<INSTALL_INTERFACE:include>
)
# ==================================================================================


# ==================================================================================
# 3. Link Dependencies
# ==================================================================================
# As foundational layer, it only depends on external libraries.
# We link system OpenMP, Eigen, and Cereal so any module linking
# trex::utils inherits them.
target_link_libraries(trex_utils PUBLIC
    Eigen3::Eigen
    cereal::cereal
    OpenMP::OpenMP_CXX
)
# ==================================================================================


# ==================================================================================
# 4. Compiler Settings
# ==================================================================================
# Require at least C++20 for the utils module
target_compile_features(trex_utils PUBLIC cxx_std_20)

target_compile_definitions(trex_utils PUBLIC
    # Only route Eigen through an external BLAS when it is available;
    # TREX_USE_BLAS_LAPACK=OFF (Python/portable builds) keeps Eigen self-contained.
    $<$<BOOL:${TREX_USE_BLAS_LAPACK}>:EIGEN_USE_BLAS>
    # MSVC: expose M_PI from <cmath>; stop <windows.h> from defining min/max macros
    $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES;NOMINMAX>
)

target_compile_options(trex_utils PUBLIC
    # Eigen/GTest-heavy TUs exceed MSVC's default COFF section limit
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
)

trex_set_compile_options(trex_utils)
# ==================================================================================


# ==================================================================================
# 4b. Public Header File Set
# ==================================================================================
target_sources(trex_utils PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/datageneration/utils_datagen.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/datageneration/utils_dummygen.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/datageneration/utils_noisegen.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/eval_metrics/utils_eval_cdiagnostics.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/eval_metrics/utils_eval_composites.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/eval_metrics/utils_eval_counts.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/eval_metrics/utils_eval_rates.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fp_classify/fp_classify.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/global_temp_dir.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/logging/logger.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/memmap/memory_mapped_matrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/openmp/utils_openmp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization/utils_cereal_armadillo.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization/utils_cereal_eigen.hpp
)
# ==================================================================================


# ==================================================================================
# 5. Subdirectories
# ==================================================================================
# Sources and headers from subfolders are appended to the target here
add_subdirectory(datageneration)
add_subdirectory(eval_metrics)
add_subdirectory(memmap)
add_subdirectory(openmp)
add_subdirectory(serialization)
# ==================================================================================


# ==================================================================================
# 6. Installation
# ==================================================================================
install(TARGETS trex_utils
    EXPORT TRexSelectorTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# ==================================================================================
