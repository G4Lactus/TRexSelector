# src/trex/CMakeLists.txt
# ===========================================================================
# CMake configuration for the trex module
# ===========================================================================


# ----- ADD SUBDIRECTORIES ---------------------------------------------

# ------ Add trex_core subdirectory ------
add_subdirectory(trex_core)               # Core T-Rex selector functionalities

# ------ Add trex_da subdirectory ------
# add_subdirectory(trex_da)               # T-Rex dependency aware functionalities

# ------ Add trex_gvs subdirectory ------
# add_subdirectory(trex_gvs)              # T-Rex grouped variable selection functionalities

# ------ Add trex_screening subdirectory ------
# add_subdirectory(trex_screening)        # T-Rex screening functionalities

# ------ Add trex_sparse_pca subdirectory ------
# add_subdirectory(trex_sparse_pca)      # T-Rex sparse PCA functionalities

# ------ END OF FILE -----------------------------------------------------
