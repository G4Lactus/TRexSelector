# src/tsolvers/solver_utils/CMakeLists.txt
# ==================================================================================
# CMakeLists.txt for the solver_utils module of the tsolvers library.
# ==================================================================================


# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_tsolvers PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/solver_preprocessing.cpp
)
# ==================================================================================


# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
# Marked PUBLIC so that CMake tools and IDEs index them correctly.
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/src      # project root src/
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/solver_preprocessing.hpp
)
# ==================================================================================
