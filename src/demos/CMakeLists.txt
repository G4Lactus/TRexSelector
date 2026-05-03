# src/demos/CMakeLists.txt
# ===================================================================================
# CMake configurations for T-Algorithm and TRex demo executables.
#
# Organization:
# -------------
# |-- src/demos/
# |-- applications
# |-- ml_methods/
# |-- trex_methods/
# |-- tsolvers/
# ===================================================================================


# ===================================================================================
# 1. Directory setup
# ===================================================================================
set(DIRSRC ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(SUBDIR "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
# ===================================================================================


# ===================================================================================
# 2. Library Sets
# ===================================================================================
# Base libraries for all demo executables.
set(BASE_LIBS

    # Core dependencies for safety (redundancy)
    Boost::headers
    Boost::iostreams

    # Linear algebra backends
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Libraries for Data Preprocessing demo
set(DATA_PREPROCESSING_LIBS
    ${BASE_LIBS}
    trex::ml_methods                    # ML tools
)

# Libraries for T-Algorithm based demos
set(T_ALGORITHM_LIBS
    ${BASE_LIBS}
    trex::ml_methods                    # ML tools
    trex::tsolvers                      # T-Algorithm solvers
)

# Libraries for TRex Selector based demos
set(T_REX_SELECTOR_LIBS
    ${BASE_LIBS}
    trex::trex_selector_methods         # T-Rex Selectors
)


# ==================================================================================
# Macro: add_demo_target
# ==================================================================================
# Creates a demo executable with proper include paths, linking, and compile flags.
#
# Usage:
#   add_demo_target(<target_name> <demo_file> <library_set> [additional_libraries...])
#
# Arguments:
#   target_name          - Name of the demo executable (e.g., demo_00_scaler)
#   demo_file            - Source file for the demo (e.g., demo_00_scaler.cpp)
#   library_set          - Library variable to use (e.g., T_ALGORITHM_LIBS, T_REX_SELECTOR_LIBS,...)
#   additional_libraries - (Optional) Additional libraries to link against
#
# Example:
#   add_demo_target(demo_01_tlars demo_01_tlars.cpp T_ALGORITHM_LIBS)
#
# ==================================================================================
macro(add_demo_target target_name demo_file library_set)

    # Create executable
    add_executable(${target_name}
        ${demo_file}
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include     # Public headers
        ${CMAKE_SOURCE_DIR}/src         # Source headers
        ${ADDITIONAL_INCLUDES}          # Any additional includes
    )

    # Link libraries
    target_link_libraries(${target_name} PRIVATE
        Eigen3::Eigen                   # Explicit redundancy for safety
        ${${library_set}}               # Core libraries for the demo type
        ${ARGN}                         # Additional libraries passed to the macro
    )

    # Compile definitions
    target_compile_definitions(${target_name} PRIVATE
        EIGEN_USE_BLAS                  # Use BLAS backend for Eigen
    )

    # Compile options (Executables need their own optimization flags)
    # NOTE: -ffast-math breaks std::isnan — see trex_selector_methods/CMakeLists.txt.
    target_compile_options(${target_name} PRIVATE
        -Wno-deprecated-declarations
        $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math -funroll-loops>
    )

    # Clear additional includes for next use
    set(ADDITIONAL_INCLUDES "")
endmacro()
# ==================================================================================


# ===================================================================================
# Demo Targets
# ===================================================================================

# ===================================================================================
# ML Methods based demos
# ===================================================================================

# Demo 01: Data preprocessing (Z-Score, Lp-Norm)
add_demo_target(demo_mlm_01_scaler
    ml_methods/demo_mlm_01_scaler.cpp
    DATA_PREPROCESSING_LIBS
)

# Demo 02: Hierarchical Clustering (CLINK, SLINK, UPGMA (NN-CHAIN))
add_demo_target(demo_mlm_02_hac
    ml_methods/demo_mlm_02_hac.cpp
    DATA_PREPROCESSING_LIBS
)

# Demo 03: HAC via Memory-Mapped I/O at biobank scale (N=100k, P=400k, K=10, ~320 GB)
add_demo_target(demo_mlm_03_hac_mmap
    ml_methods/demo_mlm_03_hac_mmap.cpp
    DATA_PREPROCESSING_LIBS
)
target_compile_definitions(demo_mlm_03_hac_mmap PRIVATE
    "TREX_DEMO_MMAP_PATH=\"${CMAKE_SOURCE_DIR}/trex_mlm_demo_03_hac_mmap.bin\""
)


# ===================================================================================
# T-Algorithm based demos
# ===================================================================================
# Lars based demos
# -----------------------------------------------------------------------------------
# Demo 01: T-LARS basic usage, serialization, memory mapping
add_demo_target(demo_ts_01_tlars
    tsolvers/demo_ts_01_tlars.cpp
    T_ALGORITHM_LIBS
)

# Demo 02: T-LASSO basic usage, serialization, memory mapping
add_demo_target(demo_ts_02_tlasso
    tsolvers/demo_ts_02_tlasso.cpp
    T_ALGORITHM_LIBS
)

# Demo 03: T-Stepwise basic usage, serialization, memory mapping
add_demo_target(demo_ts_03_tstepwise
    tsolvers/demo_ts_03_tstepwise.cpp
    T_ALGORITHM_LIBS
)

# Demo 04: T-ENET basic usage, serialization, memory mapping
add_demo_target(demo_ts_04_tenet
    tsolvers/demo_ts_04_tenet.cpp
    T_ALGORITHM_LIBS
)

# Demo 05: T-Stagewise basic usage, serialization, memory mapping
add_demo_target(demo_ts_05_tstagewise
    tsolvers/demo_ts_05_tstagewise.cpp
    T_ALGORITHM_LIBS
)


# -----------------------------------------------------------------------------------
# OMP based demos
# -----------------------------------------------------------------------------------
# Demo 06: T-OMP basic usage, serialization, memory mapping
add_demo_target(demo_ts_06_tomp
    tsolvers/demo_ts_06_tomp.cpp
    T_ALGORITHM_LIBS
)

# Demo 07: T-GP basic usage, serialization, memory mapping
add_demo_target(demo_ts_07_tgp
    tsolvers/demo_ts_07_tgp.cpp
    T_ALGORITHM_LIBS
)

# Demo 08: T-ACGP basic usage, serialization, memory mapping
add_demo_target(demo_ts_08_tacgp
    tsolvers/demo_ts_08_tacgp.cpp
    T_ALGORITHM_LIBS
)

# Demo 09: T-MP basic usage, serialization, memory mapping
add_demo_target(demo_ts_09_tmp
    tsolvers/demo_ts_09_tmp.cpp
    T_ALGORITHM_LIBS
)

# Demo 10: T-NCGMP basic usage, serialization, memory mapping
add_demo_target(demo_ts_10_tncgmp
    tsolvers/demo_ts_10_tncgmp.cpp
    T_ALGORITHM_LIBS
)

# Demo 11: T-OLS basic usage, serialization, memory mapping
add_demo_target(demo_ts_11_tools
    tsolvers/demo_ts_11_tools.cpp
    T_ALGORITHM_LIBS
)

# Demo 12: T-AFS basic usage, serialization, memory mapping
add_demo_target(demo_ts_12_tafs
    tsolvers/demo_ts_12_tafs.cpp
    T_ALGORITHM_LIBS
)


# ===================================================================================
# Trex Selector based demos
# ===================================================================================

# --- trex/ subfolder --- (classic T-Rex) ----------------------------------------

# Demo 01: TRex Selector with in-memory processing
set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex_selector_methods/trex)
add_demo_target(demo_trex_01_classic_trex_inmem
    trex_selector_methods/trex/demo_trex_01_classic_trex_inmem.cpp
    T_REX_SELECTOR_LIBS
)

# Demo 02: TRex Selector with memory-mapped workflows (single runs)
add_demo_target(demo_trex_02_classic_trex_mmap
    trex_selector_methods/trex/demo_trex_02_classic_trex_mmap.cpp
    T_REX_SELECTOR_LIBS
)

# Demo 03: Classic TRex — memory-mapped Monte Carlo simulations (Demo C & D)
add_demo_target(demo_trex_03_classic_trex_mmap_mc_sim
    trex_selector_methods/trex/demo_trex_03_classic_trex_mmap_mc_sim.cpp
    T_REX_SELECTOR_LIBS
)


# --- trex_screening/ subfolder --- (Screen-TRex) --------------------------------

## Demo scr_01: Screen-TRex Selector — in-memory
add_demo_target(demo_trex_scr_01_screen_trex
    trex_selector_methods/trex_screening/demo_trex_scr_01_screen_trex.cpp
    T_REX_SELECTOR_LIBS
)

## Demo scr_02: Screen-TRex Selector — memory-mapped D files
add_demo_target(demo_trex_scr_02_screen_trex_mmap
    trex_selector_methods/trex_screening/demo_trex_scr_02_screen_trex_mmap.cpp
    T_REX_SELECTOR_LIBS
)

## Demo scr_03: Screen-TRex DA on correlated data (AR1 + equi DGPs)
add_demo_target(demo_trex_scr_03_screen_trex_correlated
    trex_selector_methods/trex_screening/demo_trex_scr_03_screen_trex_correlated.cpp
    T_REX_SELECTOR_LIBS
)

## Demo scr_04: Biobank Screen-TRex Selector (Algorithm 1) — in-memory D files
add_demo_target(demo_trex_scr_04_biobank_screen_trex_inmem
    trex_selector_methods/trex_screening/demo_trex_scr_04_biobank_screen_trex_inmem.cpp
    T_REX_SELECTOR_LIBS
)

## Demo scr_05: Biobank Screen-TRex Selector (Algorithm 1) — memory-mapped D files
add_demo_target(demo_trex_scr_05_biobank_screen_trex_mmap
    trex_selector_methods/trex_screening/demo_trex_scr_05_biobank_screen_trex_mmap.cpp
    T_REX_SELECTOR_LIBS
)

## Demo scr_06: Screen-TRex solver comparison (TLARS / TAFS-0.3 / TOMP)
add_demo_target(demo_trex_scr_06_screen_trex_solvers
    trex_selector_methods/trex_screening/demo_trex_scr_06_screen_trex_solvers.cpp
    T_REX_SELECTOR_LIBS
)


# --- trex_da/ subfolder --- (DA-TRex) -------------------------------------------

## Demo da_01: AR(1) Toeplitz — single-run + MC SNR sweep
add_demo_target(demo_trex_da_01_ar1
    trex_selector_methods/trex_da/demo_trex_da_01_ar1.cpp
    T_REX_SELECTOR_LIBS
)

## Demo da_02: Equicorrelation — single-run + MC SNR sweep
add_demo_target(demo_trex_da_02_equi_and_block_equi
    trex_selector_methods/trex_da/demo_trex_da_02_equi_and_block_equi.cpp
    T_REX_SELECTOR_LIBS
)

## Demo da_03: Nearest Neighbours (MA(kappa)) — single-run + MC sweeps
add_demo_target(demo_trex_da_03_nn
    trex_selector_methods/trex_da/demo_trex_da_03_nn.cpp
    T_REX_SELECTOR_LIBS
)

## Demo da_03b: DA+NN on AR(1) data — single-run + MC sweeps + 2D SNR×rho
add_demo_target(demo_trex_da_03b_nn_ar
    trex_selector_methods/trex_da/demo_trex_da_03b_nn_ar.cpp
    T_REX_SELECTOR_LIBS
)

## Demo da_04: AR(1)-block BT sweeps — 4-part sweep (SNR, rho, Q, M) (R05)
add_demo_target(demo_trex_da_04_bt_ar1_block
    trex_selector_methods/trex_da/demo_trex_da_04_bt_ar1_block.cpp
    T_REX_SELECTOR_LIBS
)

## Demo da_05: AR(1)+white-noise block BT sweeps — 4-part sweep (SNR, rho, Q, M)
add_demo_target(demo_trex_da_05_ar1_block_sweeps
    trex_selector_methods/trex_da/demo_trex_da_05_ar1_block_sweeps.cpp
    T_REX_SELECTOR_LIBS
)

## Demo da_06: Heavy-tailed block BT sweeps — 6-part sweep (SNR, rho, Q, M, tFDR, linkage)
add_demo_target(demo_trex_da_06_heavy_tailed_block_sweeps
    trex_selector_methods/trex_da/demo_trex_da_06_heavy_tailed_block_sweeps.cpp
    T_REX_SELECTOR_LIBS
)

## Demo da_07: Heavy-tailed+white-noise block BT sweeps — 5-part sweep (SNR, rho, Q, M, tFDR)
add_demo_target(demo_trex_da_07_ht_block_white
    trex_selector_methods/trex_da/demo_trex_da_07_ht_block_white.cpp
    T_REX_SELECTOR_LIBS
)

## Demo da_08: Prior Groups — single-run + MC SNR sweep
add_demo_target(demo_trex_da_08_groups
    trex_selector_methods/trex_da/demo_trex_da_08_groups.cpp
    T_REX_SELECTOR_LIBS
)


# --- trex_gvs/ subfolder --- (GVS-TRex) -------------------------------------------

## Demo gvs_01: TRex+GVS — basic block-AR(1) demo (IEN + EN)
add_demo_target(demo_trex_gvs_01_basic
    trex_selector_methods/trex_gvs/demo_trex_gvs_01_basic.cpp
    T_REX_SELECTOR_LIBS
)


# ===================================================================================
# Matplot++ based demos
# ===================================================================================
## Demo 11: Plot dummy matrices
# set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex/trex_core)
# add_demo_target(demo_11_plot_dummy_matrices
#    demo_11_plot_dummy_matrices.cpp
#    T_REX_SELECTOR_LIBS
#    Matplot++::matplot
# )
#
#
# set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex/trex_core)
# add_demo_target(demo_12_biodata
#    demo_12_biodata.cpp
#    T_REX_SELECTOR_LIBS
#    Matplot++::matplot
# )
#
#
# set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex/trex_core)
# add_demo_target(demo_13_plot_fdr_tpr_results
#    demo_13_plot_fdr_tpr_results.cpp
#    T_REX_SELECTOR_LIBS
#    Matplot++::matplot
# )
#
#
# Demo 14: TRex Selector with HapGen Data
# ===================================================================================
# set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex/trex_core)
# add_demo_target(demo_14_trex_hapgen
#     demo_14_trex_hapgen.cpp
#     T_REX_SELECTOR_LIBS
#     Matplot++::matplot
# )
#
