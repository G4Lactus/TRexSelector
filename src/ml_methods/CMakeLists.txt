# src/ml_methods/CMakeLists.txt
# ==================================================================================
# CMake configuration for the ml_methods module.
#
# This library provides preprocessing methods, machine learning algorithms, and
# machine learning utilities for the T-Rex framework.
#
# Organization:
# -------------
# src/ml_methods/
# |-- standardization
# |-- clustering
# |-- model_selection
# ==================================================================================


# ==================================================================================
# 1. Library Target
# ==================================================================================
# Create a static library for ML methods (sources will be appended by subdirs)
add_library(trex_ml_methods STATIC "")

# Add alias for consistent naming (mirrors C++ namespace)
add_library(trex::ml_methods ALIAS trex_ml_methods)
# ==================================================================================


# ==================================================================================
# 2. Include Directories
# ==================================================================================
target_include_directories(trex_ml_methods PUBLIC
    # Build-tree
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>    # Public API headers
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>        # Internal headers

    # Install-tree
    $<INSTALL_INTERFACE:include>                       # Installed headers
    $<INSTALL_INTERFACE:src>                           # Append to mirror build-tree structure
)
# ==================================================================================


# ==================================================================================
# 3. Link Dependencies
# ==================================================================================
target_link_libraries(trex_ml_methods PUBLIC
    # Internal Dependencies
    trex::utils

    # External Dependencies
    Eigen3::Eigen
    cereal::cereal
)
# ==================================================================================


# ==================================================================================
# 4. Compiler Settings
# ==================================================================================
# Require  at least C++20 strictly for this module
target_compile_features(trex_ml_methods PUBLIC cxx_std_20)

target_compile_definitions(trex_ml_methods PUBLIC
    EIGEN_USE_BLAS               # Use BLAS backend in Eigen
)

target_compile_options(trex_ml_methods PRIVATE
    -Wno-deprecated-declarations
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math -funroll-loops>
)
# ==================================================================================


# ==================================================================================
# 5. Include Subdirectories
# ==================================================================================
# Sources and headers from subfolders are appended to target
add_subdirectory(standardization)
add_subdirectory(clustering)
add_subdirectory(model_selection)
# ==================================================================================
