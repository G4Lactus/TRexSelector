# src/utils/memmap/CMakeLists.txt
# ==================================================================================
# CMake configuration for the memmap utility module.
#
# This module provides memory-mapped matrix functionalities using Boost.
#
# Organization:
# -------------
# |-- utils/memmap/
# |- memory_mapped_matrix.hpp/.cpp
# ==================================================================================


# ==================================================================================
# 1. Link Dependencies
# ==================================================================================
# Append specific Boost dependencies to the parent library target (trex_utils).
# Eigen is already linked at the master utils level, so we don't need to repeat it.
target_link_libraries(trex_utils PUBLIC
    Boost::headers
    Boost::iostreams
)
# ==================================================================================


# ==================================================================================
# 2. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_utils PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/memory_mapped_matrix.cpp
)
# ==================================================================================


# ==================================================================================
# 3. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_utils PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/memory_mapped_matrix.hpp
)
# ==================================================================================
