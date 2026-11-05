# src/tsolvers/CMakeLists.txt
# ==============================================================================
# CMake configuration for T-Rex solver library (T-LARS family algorithms).
#
# This library provides algorithms for the terminating variable selection
# framework used within the T-Rex Selector.
#
# The following solvers are implemented:
#
#   LARS (Least Angle Regression) path based solvers:
#   - TLARS:      Terminating Least Angle Regression Solver (LARS).
#   - TLASSO:     T-LARS with L1 penalty and variable removal (LASSO).
#   - TSTEPWISE:  Terminating Forward Stepwise Regression Solver (LARS based FS).
#   - TSTAGEWISE: Terminating Forward Stagewise Regression Solver (LARS based Stagewise).
#   - TENET:      Terminating Elastic Net Solver (L1 + L2 penalty, Pathwise algorithm, LARS-EN).
#   - TIENET:     Terminating Informed Elastic Net Solver (L1 + L2 penalty, Pathwise algorithm,
#                 LARS-IEN).
#
#
#   OMP (Orthogonal Matching Pursuit) based solvers:
#   - TOMP:      Terminating Orthogonal Matching Pursuit Solver (OMP).
#   - TGP:       Terminating Gradient Pursuit Solver (GP).
#   - TACGP:     Terminating Approximate Conjugate Gradient Pursuit Solver (ACGP).
#   - TMP:       Terminating Matching Pursuit Solver (MP).
#
#
#   AFS (Adaptive Forward Selection) based solvers:
#   - TAFS:      Terminating Adaptive Forward Selection Solver (AFS).
#
#
#   VD (Virtual Dummy) forward selectors (dummies realized on demand):
#   - VD_LARS:         Virtual Dummy LARS (equiangular direction, gamma-step).
#   - VD_OMP:          Virtual Dummy OMP (greedy argmax, full OLS refit).
#   - VD_AFS:          Virtual Dummy Adaptive Forward Stepwise.
#   - VD_AFS_Logistic: VD-AFS with logistic link (binary response).
#
#
#   SD (Sparse Dummy) terminating solvers (sparse Rademacher dummies,
#   inner products replaced by index-set sums):
#   - SD_TLARS:  T-LARS racing X against an on-demand sparse-dummy pool.
#
#
# Organization (one peer folder per dummy mechanism; the three families share
# no code, and no class serves more than one mechanism):
# ----------------------------------------------------------------------------
# src/tsolvers/
# |-- tsolvers.hpp              umbrella of the classical family (= ad.hpp)
# |-- ad.hpp  vd.hpp  sd.hpp    one umbrella per family
# |
# |-- ad/                       augmented (appended) dummies: solvers run on [X, D]
#     |-- tsolver_base.{hpp,cpp}, solver_options.hpp, solver_utils/
#     |-- linear_model/{lars_based, omp_based, afs_based, cd_based}/
# |
# |-- vd/                       virtual dummies, realized on demand (VD_Base root)
# |
# |-- sd/                       sparse virtual dummies (SDTSolver_Base root)
#
# Placement rule: a file lives at the level of its lowest common user; it is
# promoted to a shared folder only when a second family needs it.
# ==============================================================================


# ==============================================================================
# 1. Library Target
# ==============================================================================
# Create a static library for tsolvers (sources are appended by the family
# subdirectories ad/, vd/, sd/)
add_library(trex_tsolvers STATIC "")

# Set target alias for consistent naming
add_library(trex::tsolvers ALIAS trex_tsolvers)
# ==============================================================================


# ==============================================================================
# 2. Include Directories
# ==============================================================================
target_include_directories(trex_tsolvers PUBLIC
    # Build-tree (dev)
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>      # Public API headers
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>          # Internal headers

    # Install-tree (user)
    $<INSTALL_INTERFACE:${TREX_INSTALL_INCLUDE_ROOT}>                        # Installed headers
)
# ==============================================================================


# ==============================================================================
# 3. Link Dependencies
# ==============================================================================
target_link_libraries(trex_tsolvers
    PUBLIC
        # Unified Utils Library (provides memmap, datagen, serialization, etc.)
        trex::utils

        # Explicit Redundancy
        Eigen3::Eigen                   # Core linear algebra
        cereal::cereal                  # Serialization
)

# OpenMP is optional (TREX_OPENMP_ENABLED, root CMakeLists.txt)
if(TREX_OPENMP_ENABLED)
    target_link_libraries(trex_tsolvers PUBLIC OpenMP::OpenMP_CXX)   # Parallel computing
endif()

# BLAS/LAPACK imported targets exist only when TREX_USE_BLAS_LAPACK is ON.
# Python and portable/no-BLAS builds disable them, so link conditionally to
# avoid referencing targets (BLAS::BLAS / LAPACK::LAPACK) that were never
# created by find_package.
if(TREX_USE_BLAS_LAPACK)
    target_link_libraries(trex_tsolvers
        PRIVATE
            BLAS::BLAS                  # BLAS backend (imported target)
            LAPACK::LAPACK              # LAPACK backend (imported target)
    )
endif()
# ==============================================================================


# ==============================================================================
# 4. Compiler Settings
# ==============================================================================
# Require at least C++20 strictly for this module
target_compile_features(trex_tsolvers PUBLIC cxx_std_20)

target_compile_definitions(trex_tsolvers PUBLIC
    # Only route Eigen through an external BLAS when it is available;
    # TREX_USE_BLAS_LAPACK=OFF (Python/portable builds) keeps Eigen self-contained.
    $<$<BOOL:${TREX_USE_BLAS_LAPACK}>:EIGEN_USE_BLAS>
)

trex_set_compile_options(trex_tsolvers)
# ==============================================================================


# ==============================================================================
# 4b. Public Header File Set
# ==============================================================================
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${TREX_INSTALL_INCLUDE_ROOT}>
    FILES
        # Umbrellas only; every family lists its own headers in its subtree.
        ${CMAKE_CURRENT_SOURCE_DIR}/tsolvers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ad.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vd.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sd.hpp
)
# ==============================================================================


# ==============================================================================
# 5. Include Subdirectories
# ==============================================================================
# One subdirectory per dummy mechanism; each appends its sources and headers.
add_subdirectory(ad)
add_subdirectory(vd)
add_subdirectory(sd)
# ==============================================================================


# ==============================================================================
# 6. Install Rules
# ==============================================================================

# Install compiled library and export the target
if(TREX_INSTALL_CPP_SDK)
install(TARGETS trex_tsolvers
    EXPORT   TRexSelectorTargets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}          # static .a
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}          # shared .so (future)
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}          # executables
    FILE_SET HEADERS                                      # installs all FILE_SET headers
             DESTINATION ${TREX_INSTALL_INCLUDE_ROOT}     # namespaced root, subfolders kept
)
endif()

# ==============================================================================
