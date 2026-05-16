# src/tsolvers/linear_model/omp_based/CMakeLists.txt
# ==============================================================================
# CMakeLists.txt for the OMP-based solvers for the linear regression model.
# ==============================================================================

# ==============================================================================
# 1. Append Source Files (.cpp - private)
# ==============================================================================
target_sources(trex_tsolvers PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/tomp_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tgp_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tacgp_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tmp_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tncgmp_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tools_solver.cpp
)
# ==============================================================================


# ==============================================================================
# 2. Include Directories (.hpp - public)
# ==============================================================================
# Marked PUBLIC so that CMake tools and IDEs index them correctly.
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/src          # project root /src
    FILES
        # OMP based solvers
        # -----------------------------------------------
        ${CMAKE_CURRENT_LIST_DIR}/tomp_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tgp_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tacgp_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tmp_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tncgmp_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tools_solver.hpp
)
# ==============================================================================
