# src/ml_methods/ridge_regression/CMakeLists.txt
# ==================================================================================
# CMake configuration for the ridge_regression module of ml_methods.
#
# Header-only module providing a standalone Ridge Regression solver.
# ==================================================================================


# ==================================================================================
# 1. Append Source Files
# ==================================================================================
# No .cpp sources — header-only module.
# ==================================================================================


# ==================================================================================
# 2. Append Header Files
# ==================================================================================
# Append header files to parent library target for IDE visibility.
target_sources(trex_ml_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/ridge.hpp
)
# ==================================================================================
