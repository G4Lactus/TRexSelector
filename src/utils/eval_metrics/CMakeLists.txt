# src/utils/eval_metrics/CMakeLists.txt
# ==================================================================================
# CMake configuration for the eval_metrics utility module
#
# This module provides evaluation metrics, counts, rates, and diagnostics for
# machine learning models.
# ==================================================================================


# ==================================================================================
# 1. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_utils PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/utils_eval_cdiagnostics.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utils_eval_composites.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utils_eval_counts.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utils_eval_rates.hpp
)
# ==================================================================================
