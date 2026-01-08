# src/ml_methods/CMakeLists.txt
# ==================================================================================
# CMake configuration for the ml_methods module
#
# This library provides preprocessing methods, machine learning algorithms, and
# machine learning utilities for the T-Rex framework.
# ==================================================================================


# ==================================================================================
# Library Target
# ==================================================================================
# Create a static library for ML methods: libtrex_ml_methods.a
add_library(trex_ml_methods STATIC
    StandardScaler.cpp
    Normalizer.cpp
)

# Add alias for consistent naming
add_library(trex::ml_methods ALIAS trex_ml_methods)
# ==================================================================================


# ==================================================================================
# Include Directories
# ==================================================================================
target_include_directories(trex_ml_methods PUBLIC
    # Build-tree
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>    # Public API headers
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>        # Internal headers
    # Install-tree
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:src>
)

# ==================================================================================
# Link Dependencies
# ==================================================================================
target_link_libraries(trex_ml_methods PUBLIC
    # Internal Dependencies
    trex::utils::openmp
    trex::utils::serialization

    # External Dependencies
    Eigen3::Eigen
    cereal::cereal
)
# ==================================================================================


# ==================================================================================
# Compiler Settings
# ==================================================================================
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
