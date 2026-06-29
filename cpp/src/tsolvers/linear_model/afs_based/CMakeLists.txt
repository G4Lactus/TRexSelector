# src/tsolvers/linear_model/afs_based/CMakeLists.txt
# ==============================================================================
# CMakeLists.txt for the AFS-based solvers for the linear regression model.
# ==============================================================================


# ==============================================================================
# 1. Append Source Files (.cpp - private)
# ==============================================================================
target_sources(trex_tsolvers PRIVATE
    # AFS based solvers
    # -----------------------------------------------
    ${CMAKE_CURRENT_LIST_DIR}/tafs_solver.cpp
)
# ==============================================================================


# ==============================================================================
# 2. Append Header Files (.hpp - public)
# ==============================================================================
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src          # project root /src
    FILES
        # AFS based solvers
        # -----------------------------------------------
        ${CMAKE_CURRENT_LIST_DIR}/tafs_solver.hpp
)
# ==============================================================================
