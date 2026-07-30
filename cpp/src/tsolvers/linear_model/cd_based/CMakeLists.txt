# src/tsolvers/linear_model/cd_based/CMakeLists.txt
# ==============================================================================
# CMakeLists.txt for the CCD-based solvers for the linear regression model.
# ==============================================================================


# ==============================================================================
# 1. Append Source Files (.cpp - private)
# ==============================================================================
target_sources(trex_tsolvers PRIVATE
    # CCD based solvers
    # -----------------------------------------------
    ${CMAKE_CURRENT_LIST_DIR}/tccd_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tcenet_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tcienet_solver.cpp
)
# ==============================================================================


# ==============================================================================
# 2. Append Header Files (.hpp - public)
# ==============================================================================
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src          # project root /src
    FILES
        # CCD based solvers
        # -----------------------------------------------
        ${CMAKE_CURRENT_LIST_DIR}/tccd_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tcenet_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tcienet_solver.hpp
)
# ==============================================================================
