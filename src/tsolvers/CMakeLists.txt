# src/tsolvers/CMakeLists.txt
# ==============================================================================
# CMake configuration for T-Rex solver library (T-LARS family algorithms)
#
# This library provides terminating variable selection algorithms for use with
# the T-Rex Selector framework.
#   - TLARS:     Terminating Least Angle Regression Solver (LARS)
#   - TLASSO:    T-LARS with L1 penalty and variable removal (LASSO)
#   - TSTEPWISE: Terminating Forward Stepwise Regression Solver (LARS based)
#   - TENET:     Terminating Elastic Net Solver (L1 + L2 penalty, Pathwise algorithm, LARS-EN)
#   - TOMP:      Terminating Orthogonal Matching Pursuit Solver (OMP)
#   - TGP:       Terminating Gradient Pursuit Solver (GP)
#   - TACGP:     Terminating Approximate Conjugate Gradient Pursuit Solver (ACGP)
#
# ==============================================================================

# ==============================================================================
# Source Files
# ==============================================================================
# .cpp files for the tsolvers module
set(TSOLVERS_SOURCES
    # Core LARS-based solvers
    TLARS_Solver.cpp
    TLASSO_Solver.cpp
    TSTEPWISE_Solver.cpp
    TENET_Solver.cpp

    # OMP-based solvers
    TOMP_Solver.cpp
    TGP_Solver.cpp
    TACGP_Solver.cpp
)

# .hpp files for the tsolvers module
set(TSOLVER_HEADERS
    # Core LARS-based solvers
    TLARS_Solver.hpp
    TLASSO_Solver.hpp
    TSTEPWISE_Solver.hpp
    TENET_Solver.hpp

    # OMP-based solvers
    TOMP_Solver.hpp
    TGP_Solver.hpp
    TACGP_Solver.hpp
)


# ==============================================================================
# Library Target
# ==============================================================================
add_library(trex_tsolvers STATIC
    ${TSOLVERS_SOURCES}
)

# Set target alias for consistent naming
add_library(trex::tsolvers ALIAS trex_tsolvers)


# ==============================================================================
# Include Directories for hierarchical includes (Generator Expressions)
# ==============================================================================
target_include_directories(trex_tsolvers PUBLIC
    # Build-tree
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>      # Public API headers
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>          # Internal headers
    # Install-tree
    $<INSTALL_INTERFACE:include>                        # Installed headers
)

# ==============================================================================
# Link Dependencies
# ==============================================================================
# Create a static library for the tsolvers module
target_link_libraries(trex_tsolvers PUBLIC
    Eigen3::Eigen                   # Core linear algebra
    OpenMP::OpenMP_CXX              # Parallel computing
    cereal::cereal                  # Serialization
    trex::utils::serialization      # Used in .hpp files
)

target_link_libraries(trex_tsolvers PRIVATE
    ${BLAS_LIBRARIES}           # BLAS backend
    ${LAPACK_LIBRARIES}         # LAPACK backend
    trex::utils::datagen        # Data generation utilities
    trex::utils::memmap         # Memory-mapped file support
)


# ==============================================================================
# Compile Features and Definitions
# ==============================================================================
target_compile_features(trex_tsolvers PUBLIC cxx_std_20)

target_compile_definitions(trex_tsolvers PUBLIC
    EIGEN_USE_BLAS              # Use BLAS backend in Eigen
)


# ==============================================================================
# Compile Options
# ==============================================================================
target_compile_options(trex_tsolvers PRIVATE
    -Wno-deprecated-declarations
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math -funroll-loops>
)


# ==============================================================================
# Installation (Optional)
# ==============================================================================
# install(TARGETS trex_tsolvers
#     EXPORT trex_tsolvers-targets
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
#     INCLUDES DESTINATION include
# )
#
# install(FILES ${TSOLVER_HEADERS}
#     DESTINATION include/tsolvers
# )
#
