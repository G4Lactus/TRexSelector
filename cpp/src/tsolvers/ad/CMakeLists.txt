# src/tsolvers/ad/CMakeLists.txt
# ==============================================================================
# Augmented-dummy (AD) T-solver family: the classical terminating solvers that
# run on the augmented design [X, D].
#
# TSolver_Base, SolverOptions and solver_utils/ live here rather than one level
# up because every user of them is an AD solver: VD_Base and SDTSolver_Base are
# standalone roots. Promote a piece to a shared folder only when a second
# dummy mechanism needs it.
# ==============================================================================


# ==============================================================================
# 1. Append Source Files (.cpp - private)
# ==============================================================================
target_sources(trex_tsolvers PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/tsolver_base.cpp
)
# ==============================================================================


# ==============================================================================
# 2. Append Header Files (.hpp - public)
# ==============================================================================
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src          # project root /src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/tsolver_base.hpp
        ${CMAKE_CURRENT_LIST_DIR}/solver_options.hpp
)
# ==============================================================================


# ==============================================================================
# 3. Subdirectories
# ==============================================================================
add_subdirectory(linear_model)
add_subdirectory(solver_utils)
# ==============================================================================
