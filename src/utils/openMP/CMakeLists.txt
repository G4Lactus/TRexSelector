# src/utils/openmp/CMakeLists.txt
# ==================================================================================
# CMake configuration for the utils/openmp module.
#
# This module provides safe threading wrappers and OpenMP utilities.
# ==================================================================================

# ==================================================================================
# 1. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_utils PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/utils_openmp.hpp
)
# ==================================================================================
