# src/demos/trex_selector_methods/trex_da/CMakeLists.txt
# ==================================================================================
# Demo executables for the Dependency-Aware T-Rex selector.
# ==================================================================================

add_demo_target(demo_trex_da_01_ar1
    demo_trex_da_01_ar1.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_02_equi_and_block_equi
    demo_trex_da_02_equi_and_block_equi.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_03_nn
    demo_trex_da_03_nn.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_03b_nn_ar
    demo_trex_da_03b_nn_ar.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_04_bt_ar1_block
    demo_trex_da_04_bt_ar1_block.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_05_ar1_block_sweeps
    demo_trex_da_05_ar1_block_sweeps.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_06_heavy_tailed_block_sweeps
    demo_trex_da_06_heavy_tailed_block_sweeps.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_07_ht_block_white
    demo_trex_da_07_ht_block_white.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_da_08_groups
    demo_trex_da_08_groups.cpp
    T_REX_SELECTOR_LIBS
)
