# src/ml_methods/model_selection/CMakeLists.txt
# ==================================================================================
# CMake configuration for the model_selection module of ml_methods.
#
# This module provides model selection utilities to be used within the T-Rex
# framework and its algorithms.
#
# Organization:
# -------------
# src/ml_methods/
# |-- model_selection
#
# ==================================================================================

# ==================================================================================
# 1. Append Source Files
# ==================================================================================
# Append the C++ source files (.cpp) to the parent library target (trex_ml_methods).
# These are marked PRIVATE because they are compiled implementation details.
target_sources(trex_ml_methods PRIVATE
    # ${CMAKE_CURRENT_LIST_DIR}/XYZ.cpp
)
# ==================================================================================


# ==================================================================================
# 2. Append Header Files
# ==================================================================================
# Append header files to parent library target for IDE visibility.
# Marked PUBLIC so that CMake tools and IDEs index them correctly for developers.
target_sources(trex_ml_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/ridge_gcv.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ridge_cv.hpp
)
# ==================================================================================
