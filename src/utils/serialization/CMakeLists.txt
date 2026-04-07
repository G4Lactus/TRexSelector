# src/utils/serialization/CMakeLists.txt
# ==================================================================================
# CMake configuration for the serialization utility module.
#
# This module provides custom Cereal serialization extensions for Eigen matrices
# and Standard Library types.
# ==================================================================================


# ==================================================================================
# 1. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_utils PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/utils_cereal_armadillo.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utils_cereal_eigen.hpp
)
# ==================================================================================
