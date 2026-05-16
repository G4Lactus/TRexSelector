# src/trex_selector_methods/utils/CMakeLists.txt
# ==================================================================================
# CMake configuration for the utils module of T-Rex selector methods.
#
# ==================================================================================

# ==================================================================================
# 1. Append Source Files (.cpp - private)
# ==================================================================================
target_sources(trex_selector_methods PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/trex_data_normalizer.cpp
)

# ==================================================================================
# 2. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_selector_methods PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/trex_data_normalizer.hpp
        ${CMAKE_CURRENT_LIST_DIR}/trex_dummy_generator.hpp
        ${CMAKE_CURRENT_LIST_DIR}/trex_experiment_runner.hpp
        ${CMAKE_CURRENT_LIST_DIR}/trex_temp_dir_manager.hpp
        ${CMAKE_CURRENT_LIST_DIR}/trex_memmap_manager.hpp
        ${CMAKE_CURRENT_LIST_DIR}/trex_solver_dispatch.hpp
        ${CMAKE_CURRENT_LIST_DIR}/trex_warm_start_manager.hpp
)
# ==================================================================================
