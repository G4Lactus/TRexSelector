# src/CMakeLists.txt
# ==============================================================================
# CMake configuration for TRexSelector source files and demo targets
# ==============================================================================

# Set source directory variable
set(DIRSRC ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(SUBDIR "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

# ------ ADD SUBDIRECTORIES ---------------------------------------------

# ------ Add demo subdirectory ------
add_subdirectory(demos)                 # Demo executables

# ------ Add ML methods subdirectory ------
add_subdirectory(ml_methods)            # Machine learning methods

# ------ Add solver libraries subdirectory ------
add_subdirectory(tsolvers)              # T-Rex solver libraries

# ------ Add utilities subdirectories ------
add_subdirectory(utils/datagen)         # Data generation utilities
add_subdirectory(utils/memmap)          # Memory-mapped matrix utility
add_subdirectory(utils/openMP)          # OpenMP compatibility layer
add_subdirectory(utils/serialization)   # Cereal serialization adapters


# ------ Optional: Build solver libraries if needed ------
# Reusable solver libraries
