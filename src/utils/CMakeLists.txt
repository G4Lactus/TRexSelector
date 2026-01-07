# src/utils/CMakeLists.txt
# ==============================================================================
# CMake configuration for TRexSelector utility source files
# ==============================================================================

# ----- Add utility subdirectories  ---------------------------------------------

# ------ Foundation utilities ------
add_subdirectory(memmap)                # Add memory-mapped matrix utility
add_subdirectory(openMP)                # Add OpenMP compatibility layer

# ------ Independent utilities ------
add_subdirectory(eval_metrics)          # Add evaluation metrics utility
add_subdirectory(serialization)         # Add serialization utility

# ------ Add data generation utility ------
add_subdirectory(datagen)

# ------ END OF FILE -----------------------------------------------------
