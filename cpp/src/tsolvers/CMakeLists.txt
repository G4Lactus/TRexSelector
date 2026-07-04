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
# Organization:
# -------------
# src/tsolvers/
# |-- linear_model/
#     |-- lars_based/
#     |-- omp_based/
#     |-- afs_based/
# |
# |-- logistic_model/
# |
# |-- poisson_model/
# |
# |-- cox_model/
# |
# |-- etc.
# ==============================================================================


# ==============================================================================
# 1. Library Target
# ==============================================================================
# Create a static library for tsolvers (sources are appended by subdirs)
add_library(trex_tsolvers STATIC "tsolver_base.cpp")

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
    $<INSTALL_INTERFACE:include>                        # Installed headers
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
        OpenMP::OpenMP_CXX              # Parallel computing
)

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

# NOTE: -ffast-math and -march=native are gated behind NOT TREX_PORTABLE_BUILD (wheel/PyPI builds).
target_compile_options(trex_tsolvers PRIVATE
    -Wno-deprecated-declarations
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -DNDEBUG -funroll-loops>
    $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${TREX_PORTABLE_BUILD}>>>:-march=native -ffast-math>
)
# ==============================================================================


# ==============================================================================
# 4b. Public Header File Set
# ==============================================================================
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/tsolvers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tsolver_base.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lm_tsolvers.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_utils/solver_preprocessing.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/lars_based.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/lars_based/tlars_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/lars_based/tlasso_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/lars_based/tenet_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/lars_based/tienet_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/lars_based/tstagewise_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/lars_based/tstepwise_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/omp_based.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/omp_based/tomp_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/omp_based/tmp_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/omp_based/tgp_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/omp_based/tncgmp_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/omp_based/tacgp_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/omp_based/tools_solver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/afs_based.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model/afs_based/tafs_solver.hpp
)
# ==============================================================================


# ==============================================================================
# 5. Include Subdirectories
# ==============================================================================
# Sources and headers from subfolders are appended to target
add_subdirectory(linear_model)
add_subdirectory(logistic_model)
add_subdirectory(solver_utils)
# Future expansion: ...
# ==============================================================================


# ==============================================================================
# 6. Install Rules
# ==============================================================================

# Install compiled library and export the target
install(TARGETS trex_tsolvers
    EXPORT   TRexSelectorTargets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}          # static .a
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}          # shared .so (future)
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}          # executables
    FILE_SET HEADERS                                      # installs all FILE_SET headers
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}      # preserves subfolder structure
)

# ==============================================================================
