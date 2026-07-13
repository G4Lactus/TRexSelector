# demos/cpp/CMakeLists.txt
# =============================================================================
# Runnable demo scenarios for the T-Rex selector variants.
#
# Included from cpp/CMakeLists.txt (option TREX_BUILD_DEMOS); binaries land
# in the build tree's bin/ next to the test executables.
# =============================================================================

# Common helper function to add a demo target
function(add_trex_demo DEMO_NAME)
    add_executable(${DEMO_NAME} ${ARGN})
    target_link_libraries(${DEMO_NAME}
        PRIVATE
            trex::utils
            trex::tsolvers
            trex::ml_methods
            trex::trex_selector_methods
    )
    target_compile_features(${DEMO_NAME} PRIVATE cxx_std_20)
    # Result files (.txt / .csv) of the MC demos land here regardless of CWD
    target_compile_definitions(${DEMO_NAME} PRIVATE
        DEMO_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/simulation_results/data/")
endfunction()

add_trex_demo(demo_trex_vd demo_trex_vd.cpp)
add_trex_demo(demo_sd_tlars demo_sd_tlars.cpp)
add_trex_demo(demo_sd2_tlars demo_sd2_tlars.cpp)
add_trex_demo(demo_sd_omp_afs demo_sd_omp_afs.cpp)
add_trex_demo(demo_trex_sd demo_trex_sd.cpp)
add_trex_demo(demo_trex_sd_mc_variable_support demo_trex_sd_mc_variable_support.cpp)
add_trex_demo(sim_trex_sd_mc sim_trex_sd_mc.cpp)
