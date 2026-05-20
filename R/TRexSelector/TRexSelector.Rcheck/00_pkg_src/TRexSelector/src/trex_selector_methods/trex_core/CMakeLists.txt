# src/trex_methods/trex/CMakeLists.txt
# ==================================================================================
# CMake configuration for the trex module.
#
# This module provides the standard T-Rex selector implementation.
# ==================================================================================


# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_selector_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/trex.cpp
)


# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/trex.hpp
)
# ==================================================================================
