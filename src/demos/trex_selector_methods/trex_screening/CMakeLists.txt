# src/demos/trex_selector_methods/trex_screening/CMakeLists.txt
# ==================================================================================
# Demo executables for the Screen T-Rex selector.
# ==================================================================================

add_demo_target(demo_trex_scr_01_screen_trex
    demo_trex_scr_01_screen_trex.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_02_screen_trex_mmap
    demo_trex_scr_02_screen_trex_mmap.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_03_screen_trex_correlated
    demo_trex_scr_03_screen_trex_correlated.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_04_biobank_screen_trex_inmem
    demo_trex_scr_04_biobank_screen_trex_inmem.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_05_biobank_screen_trex_mmap
    demo_trex_scr_05_biobank_screen_trex_mmap.cpp
    T_REX_SELECTOR_LIBS
)
add_demo_target(demo_trex_scr_06_screen_trex_solvers
    demo_trex_scr_06_screen_trex_solvers.cpp
    T_REX_SELECTOR_LIBS
)
