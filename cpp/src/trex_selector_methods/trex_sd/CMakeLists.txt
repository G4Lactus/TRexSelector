# src/trex_selector_methods/trex_sd/CMakeLists.txt
# ==================================================================================
# CMake configuration for the trex_sd module.
#
# This module provides the Sparse-Dummy T-Rex implementation (TRexSD):
# K-experiment orchestration over the SD solver family with the T-Rex
# (v*, T*) calibration (statistical machinery kept in parity with trex_vd).
# ==================================================================================


# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_selector_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/trex_sd.cpp
)
# ==================================================================================


# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/trex_sd.hpp
)
# ==================================================================================
