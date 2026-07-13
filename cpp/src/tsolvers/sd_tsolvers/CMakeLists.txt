# src/tsolvers/sd_tsolvers/CMakeLists.txt
# ==============================================================================
# CMakeLists.txt for the sparse-dummy (SD) terminating solvers.
#
# SD solvers race the original features against sparse Rademacher dummies
# (k pairs of +1/-1 entries, 2k ~= rho_d * n non-zeros per dummy). A dummy is
# represented only by its two index sets, so its inner products against the
# residual reduce to index-set sums (no multiply-adds, no dummy matrix).
# Only winning dummies are materialized into a small n x T_stop cache.
# ==============================================================================


# ==============================================================================
# 1. Append Source Files (.cpp - private)
# ==============================================================================
target_sources(trex_tsolvers PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/sd_tlars_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sd2_tlars_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sd_tomp_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sd2_tomp_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sd_tafs_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sd2_tafs_solver.cpp
)
# ==============================================================================


# ==============================================================================
# 2. Append Header Files (.hpp - public)
# ==============================================================================
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src          # project root /src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/sd_tsolver_base.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sd_tlars_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sd2_tlars_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sd_tomp_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sd2_tomp_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sd_tafs_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sd2_tafs_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sd_calibration.hpp
)
# ==============================================================================
