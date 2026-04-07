# src/CMakeLists.txt
# ==============================================================================
# CMake configuration for TRexSelector source files and demo targets.
# ==============================================================================


# ------ ADD SUBDIRECTORIES ---------------------------------------------

# ------ Add Foundation modules (orchestration) ------
add_subdirectory(utils)                     # Utilities (memmap, openmp, serialization, etc.)

# ------ Core libraries (depend on utils) -----
add_subdirectory(ml_methods)                # Core machine learning methods
add_subdirectory(tsolvers)                  # T-Solver libraries based on LARS, OMP, etc.

# ------ High-level modules (depend on tsolvers, ml_methods, utils) -----
add_subdirectory(trex_selector_methods)     # T-Rex selector libraries

# ------ Applications (depend on everything above) ------
add_subdirectory(demos)                     # Demo applications

# ------ END OF FILE -----------------------------------------------------
