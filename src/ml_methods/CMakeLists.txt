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
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>    # Public API headers
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>        # Internal headers

    # Install-tree
    $<INSTALL_INTERFACE:include>                       # Installed headers
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

# NOTE: -ffast-math breaks std::isnan — see trex_selector_methods/CMakeLists.txt.
target_compile_options(trex_ml_methods PRIVATE
    -Wno-deprecated-declarations
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math -funroll-loops>
)
# ==================================================================================


# ==================================================================================
# 4b. Public Header File Set
# ==================================================================================
target_sources(trex_ml_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/agglomerative_dispatcher.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/agglomerative_types.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/block_tiled_matrix_policy.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/dendrogram_utils.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/distance_policy.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/generic_linkage_core.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/geometric_update_policy.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/nnchain_core.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/proj_geom_updates.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/clustering/hierarchical/agglomerative/slink_core.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/model_selection/ridge_cv.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/model_selection/ridge_gcv.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/standardization/data_transformer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/standardization/lp_norm_scaler.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/standardization/z_score_scaler.hpp
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


# ==================================================================================
# 6. Installation
# ==================================================================================
install(TARGETS trex_ml_methods
    EXPORT   TRexSelectorTargets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# ==================================================================================
