# src/trex/trex_sparse_pca/CMakeLists.txt
# ==================================================================================
# CMake configuration for the trex_sparse_pca module.
#
# This module provides the T-Rex Sparse PCA implementation.
# ==================================================================================


# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_selector_methods PRIVATE
#    ${CMAKE_CURRENT_LIST_DIR}/trex_sparse_pca.cpp
)
# ==================================================================================


# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
#        ${CMAKE_CURRENT_LIST_DIR}/trex_sparse_pca.hpp
)
# ==================================================================================
