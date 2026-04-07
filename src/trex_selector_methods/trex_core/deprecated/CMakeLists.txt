# src/trex_methods/trex_core/CMakeLists.txt
# ==================================================================================
# CMake configuration for the trex_core module.
#
# This module provides the standard T-Rex Selector implementation.
# ==================================================================================


# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_selector_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/TRex_Selector.cpp
)


# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/TRex_Selector.hpp
)
# ==================================================================================
