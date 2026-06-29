# src/ml_methods/scaler_methods/CMakeLists.txt
# ==================================================================================
# CMake configuration for the scaler_methods module of ml_methods.
#
# This module provides data standardization and normalization methods for the T-Rex
# framework:
#
# Organization:
# -------------
# src/ml_methods/
# |-- scaler_methods
#     |-- data_transformer.hpp
#     |-- lp_norm_scaler.hpp/.cpp
#     |-- z_score_scaler.hpp/.cpp
# ==================================================================================

# ==================================================================================
# 1. Append Source Files
# ==================================================================================
# Append the C++ source files (.cpp) to the parent library target (trex_ml_methods).
# These are marked PRIVATE because they are compiled implementation details.
target_sources(trex_ml_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/lp_norm_scaler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/z_score_scaler.cpp
)
# ==================================================================================


# ==================================================================================
# 2. Append Header Files
# ==================================================================================
# Append header files to parent library target for IDE visibility.
# Marked PUBLIC so that CMake tools and IDEs index them correctly for developers.
target_sources(trex_ml_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/data_transformer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/lp_norm_scaler.hpp
    ${CMAKE_CURRENT_LIST_DIR}/z_score_scaler.hpp
)
# ==================================================================================
