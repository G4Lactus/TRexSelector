# src/demos/trex_selector_methods/trex/CMakeLists.txt
# ==================================================================================
# Demo executables for the classical T-Rex selector.
# ==================================================================================

add_demo_target(demo_trex_01_classic_trex_inmem
    demo_trex_01_classic_trex_inmem.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_02_classic_trex_mmap
    demo_trex_02_classic_trex_mmap.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_03_classic_trex_mmap_mc_sim
    demo_trex_03_classic_trex_mmap_mc_sim.cpp
    T_REX_SELECTOR_LIBS
)
