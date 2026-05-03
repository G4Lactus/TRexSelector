# src/trex/trex_gvs/CMakeLists.txt
# ==================================================================================
# CMake configuration for the trex_gvs module.
#
# This module provides the GVS T-Rex implementation.
# ==================================================================================


# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_selector_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/trex_gvs.cpp
)
# ==================================================================================


# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/trex_gvs.hpp
)
# ==================================================================================
