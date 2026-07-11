# src/tsolvers/vd_tsolvers/CMakeLists.txt
# ==============================================================================
# CMakeLists.txt for the virtual-dummy (VD) forward selectors.
#
# VD solvers never materialize dummy columns; each dummy is represented by
# its sufficient statistics and gets realized on demand inside the solver.
# Ported from the virtual-dummies project (Koka, Machkour, Muma).
# ==============================================================================


# ==============================================================================
# 1. Append Source Files (.cpp - private)
# ==============================================================================
target_sources(trex_tsolvers PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/vd_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vd_lars.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vd_omp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vd_afs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vd_afs_logistic.cpp
)
# ==============================================================================


# ==============================================================================
# 2. Append Header Files (.hpp - public)
# ==============================================================================
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src          # project root /src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/vd_common.hpp
        ${CMAKE_CURRENT_LIST_DIR}/vd_base.hpp
        ${CMAKE_CURRENT_LIST_DIR}/vd_lars.hpp
        ${CMAKE_CURRENT_LIST_DIR}/vd_omp.hpp
        ${CMAKE_CURRENT_LIST_DIR}/vd_afs.hpp
        ${CMAKE_CURRENT_LIST_DIR}/vd_afs_logistic.hpp
)
# ==============================================================================
