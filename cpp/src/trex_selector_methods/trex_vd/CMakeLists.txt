# src/trex_selector_methods/trex_vd/CMakeLists.txt
# ==================================================================================
# CMake configuration for the trex_vd module.
#
# This module provides the Virtual-Dummy T-Rex implementation (TRexVD).
# ==================================================================================


# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_selector_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/trex_vd.cpp
)
# ==================================================================================


# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/trex_vd.hpp
)
# ==================================================================================
