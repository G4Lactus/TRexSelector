# src/trex_selector_methods/ad/CMakeLists.txt
# ==================================================================================
# Augmented-dummy (AD) T-Rex selectors: the classical T-Rex and its DA, GVS,
# screening, SPCA and Tikhonov variants, all running the T-solvers on the
# augmented design [X, D].
#
# trex_utils/ lives here rather than one level up because every header in it
# (dummy generator, solver dispatch, warm-start manager, cluster dummies,
# normalizer, memmap manager, experiment runner) has only AD users. Promote a
# piece to a shared folder only when a second dummy mechanism needs it.
# ==================================================================================

add_subdirectory(trex_utils)
add_subdirectory(trex_core)
add_subdirectory(trex_da)
add_subdirectory(trex_gvs)
add_subdirectory(trex_screening)
add_subdirectory(trex_spca)
add_subdirectory(trex_tikhonov)
