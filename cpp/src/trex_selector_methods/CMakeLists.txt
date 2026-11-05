# src/trex_selector_methods/CMakeLists.txt
# ==================================================================================
# CMake configuration for the T-Rex selector framework module.
# Definer file. Populator files are defined in subdirectories.
#
# This library contains the T-Rex selector variants, organized as one peer
# folder per dummy mechanism (the three families share no code, and no class
# serves more than one mechanism):
#
#   ad/   augmented (appended) dummies, solvers run on [X, D]:
#         - Classical T-Rex                (trex_core)
#         - Dependency Aware T-Rex         (trex_da)
#         - Screen T-Rex, Biobank T-Rex    (trex_screening)
#         - Grouped Variable Selection     (trex_gvs)
#         - Sparse PCA T-Rex               (trex_spca)
#         - General-Tikhonov T-Rex         (trex_tikhonov)
#         - trex_utils (dummy generator, dispatch, normalizer, ...)
#   vd/   virtual dummies, realized on demand:
#         - Virtual-Dummy T-Rex            (trex_vd)
#   sd/   sparse virtual dummies:
#         - Sparse-Dummy T-Rex             (trex_sd)
#
# Placement rule: a file lives at the level of its lowest common user; it is
# promoted to a shared folder only when a second family needs it.
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
    $<INSTALL_INTERFACE:${TREX_INSTALL_INCLUDE_ROOT}>
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
    # Only route Eigen through an external BLAS when it is available;
    # TREX_USE_BLAS_LAPACK=OFF (Python/portable builds) keeps Eigen self-contained.
    $<$<BOOL:${TREX_USE_BLAS_LAPACK}>:EIGEN_USE_BLAS>
)

# All correlation sentinels use AUTO_ESTIMATE_CORRELATION (−2.0) instead of NaN.
# If you add new NaN/isnan usage, use a numeric sentinel or __builtin_isnan.
trex_set_compile_options(trex_selector_methods)
# ==================================================================================


# ==================================================================================
# 4b. Public Header File Set
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${TREX_INSTALL_INCLUDE_ROOT}>
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_core/trex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_da/trex_da.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_gvs/trex_gvs.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_screening/trex_screening.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_screening/trex_biobank_screening.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_spca/trex_spca.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_tikhonov/trex_tikhonov.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_tikhonov/tik_lambda2_pick.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_tikhonov/tik_lambda2_power_rule.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_utils/trex_cluster_dummies.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_utils/trex_cluster_hac.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_utils/trex_data_normalizer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_utils/trex_dummy_generator.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_utils/trex_experiment_runner.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_utils/trex_memmap_manager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_utils/trex_solver_dispatch.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_utils/trex_temp_dir_manager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad/trex_utils/trex_warm_start_manager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vd/trex_vd/trex_vd.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sd/trex_sd/trex_sd.hpp
)
# ==================================================================================


# ==================================================================================
# 5. Subdirectories
# ==================================================================================
# One subdirectory per dummy mechanism; each appends its sources and headers.
add_subdirectory(ad)
add_subdirectory(vd)
add_subdirectory(sd)
# ==================================================================================


# ==================================================================================
# 6. Installation
# ==================================================================================
if(TREX_INSTALL_CPP_SDK)
install(TARGETS trex_selector_methods
    EXPORT   TRexSelectorTargets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS DESTINATION ${TREX_INSTALL_INCLUDE_ROOT}
)
endif()
# ==================================================================================
