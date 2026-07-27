# src/trex_selector_methods/trex_tikhonov/CMakeLists.txt
# ==================================================================================
# CMake configuration for the trex_tikhonov module.
#
# This module provides the general-Tikhonov T-Rex implementation
# (TRexTikhonovSelector): TCIENET sparse-K dispatch through the base
# machinery with a user-supplied penalty matrix K = Gamma^T Gamma.
# ==================================================================================


# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_selector_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/trex_tikhonov.cpp
)
# ==================================================================================


# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/trex_tikhonov.hpp
)
# ==================================================================================
