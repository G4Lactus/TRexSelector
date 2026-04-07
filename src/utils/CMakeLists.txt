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
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>

    # Install-tree
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:src>
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
    EIGEN_USE_BLAS
)

target_compile_options(trex_utils PRIVATE
    -Wno-deprecated-declarations
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math -funroll-loops>
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
