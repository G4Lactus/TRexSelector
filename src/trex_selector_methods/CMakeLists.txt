# src/trex_selector_methods/CMakeLists.txt
# ==================================================================================
# CMake configuration for the T-Rex selector framework module.
# Definer file. Populator files are defined in subdirectories.
#
# This library contains the following T-Rex selector variants:
# - Classical T-Rex
# - Dependency Aware T-Rex
# - Screen T-Rex
# - Grouped Variable Selection T-Rex
# - Sparse PCA T-Rex
#
# ==================================================================================


# ==================================================================================
# 1. Library Target
# ==================================================================================
# Create a unified static library for all TRex selectors
add_library(trex_selector_methods STATIC "")

# Add alias for consistent naming (mirrors C++ namespace)
add_library(trex::trex_selector_methods ALIAS trex_selector_methods)
# ==================================================================================


# ==================================================================================
# 2. Include Directories
# ==================================================================================
target_include_directories(trex_selector_methods PUBLIC
    # Build-tree
    # ------------
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>

    # Install-tree
    # ------------
    $<INSTALL_INTERFACE:include>
)
# ==================================================================================


# ==================================================================================
# 3. Link Dependencies
# ==================================================================================
# The TRex framework sits at the top of the stack and relies on all lower modules.
target_link_libraries(trex_selector_methods PUBLIC
    trex::tsolvers     # Required for the underlying T-LARS algorithms
    trex::ml_methods   # Required for standardizing and clustering data
    trex::utils        # Inherits Eigen, OpenMP, Boost, and Cereal
)
# ==================================================================================


# ==================================================================================
# 4. Compiler Settings
# ==================================================================================
target_compile_features(trex_selector_methods PUBLIC cxx_std_20)

target_compile_definitions(trex_selector_methods PUBLIC
    EIGEN_USE_BLAS
)

# NOTE: -ffast-math implies -ffinite-math-only, which breaks std::isnan/isfinite.
# All correlation sentinels use AUTO_ESTIMATE_CORRELATION (−2.0) instead of NaN.
# If you add new NaN/isnan usage, use a numeric sentinel or __builtin_isnan.
target_compile_options(trex_selector_methods PRIVATE
    -Wno-deprecated-declarations
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -DNDEBUG -ffast-math -funroll-loops>
    $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${TREX_PORTABLE_BUILD}>>>:-march=native>
)
# ==================================================================================


# ==================================================================================
# 4b. Public Header File Set
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_core/trex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_da/trex_da.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_gvs/trex_gvs.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_screening/trex_screening.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_screening/trex_biobank_screening.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_sparse_pca/trex_sparse_pca.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_utils/trex_data_normalizer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_utils/trex_dummy_generator.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_utils/trex_experiment_runner.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_utils/trex_memmap_manager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_utils/trex_solver_dispatch.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_utils/trex_temp_dir_manager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/trex_utils/trex_warm_start_manager.hpp
)
# ==================================================================================


# ==================================================================================
# 5. Subdirectories
# ==================================================================================
add_subdirectory(trex_utils)
add_subdirectory(trex_core)
add_subdirectory(trex_da)
add_subdirectory(trex_gvs)
add_subdirectory(trex_screening)
add_subdirectory(trex_sparse_pca)
# ==================================================================================


# ==================================================================================
# 6. Installation
# ==================================================================================
install(TARGETS trex_selector_methods
    EXPORT   TRexSelectorTargets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# ==================================================================================
