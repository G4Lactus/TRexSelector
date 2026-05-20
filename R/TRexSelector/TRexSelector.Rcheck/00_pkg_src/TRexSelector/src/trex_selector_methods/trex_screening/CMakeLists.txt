# src/trex/trex_screening/CMakeLists.txt
# ==================================================================================
# CMake configuration for the trex_screening module.
#
# This module provides the Screen T-Rex implementation.
# ==================================================================================


# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_selector_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/trex_screening.cpp
    ${CMAKE_CURRENT_LIST_DIR}/trex_biobank_screening.cpp
)
# ==================================================================================


# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/trex_screening.hpp
        ${CMAKE_CURRENT_LIST_DIR}/trex_biobank_screening.hpp
)
# ==================================================================================
