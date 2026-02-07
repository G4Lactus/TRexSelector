# src/demos/CMakeLists.txt
# ===================================================================================
# CMake configurations for T-Algorithm and TRex demo executables
# ===================================================================================

# ===================================================================================
# Directory setup
# ===================================================================================
set(DIRSRC ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(SUBDIR "${CMAKE_CURRENT_SOURCE_DIR}" NAME)


# ===================================================================================
# Common libraries for different demo types
# ===================================================================================
# Base libraries for all demo executables
set(BASE_LIBS

    # Core dependencies not covered above
    Boost::headers
    Boost::iostreams

    # Linear algebra backends
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Libraries for Data Preprocessing demo
set(DATA_PREPROCESSING_LIBS
    ${BASE_LIBS}
    trex::utils::datagen                # Data generation utilities
    trex::ml_methods                    # ML tools
    trex::utils::serialization          # Serialization utilities
)

# Libraries for T-Algorithm based demos
set(T_ALGORITHM_LIBS
    ${BASE_LIBS}
    trex::utils::datagen                # Data generation utilities
    trex::ml_methods                    # ML tools
    trex::tsolvers                      # T-Algorithm solvers
)

# Libraries for TRex Selector based demos
set(T_REX_SELECTOR_LIBS
    ${BASE_LIBS}
    trex::utils::datagen                # Data generation utilities
    trex::core                          # T-Rex Selector core
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
        Eigen3::Eigen
        ${${library_set}}               # Core libraries for the demo type
        ${ARGN}                         # Additional libraries passed to the macro
    )

    # Compile definitions and options
    target_compile_definitions(${target_name} PRIVATE
        EIGEN_USE_BLAS      # Use BLAS backend for Eigen
    )

    # Compile options
    target_compile_options(${target_name} PRIVATE
        -Wno-deprecated-declarations
        $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math -funroll-loops>
    )

    # Clear additional includes for next use
    set(ADDITIONAL_INCLUDES "")
endmacro()
# ------ End of macro ------


# ===================================================================================
# Demo Targets
# ===================================================================================

# ML Methods based demos
# ===================================================================================

# Demo 00: Data preprocessing (StandardScaler, Normalizer)
add_demo_target(demo_00_scaler
    demo_00_scaler.cpp
    DATA_PREPROCESSING_LIBS
)

## T-Algorithm based demos
# ==================================================================================

# Demo 01: T-LARS basic usage, serialization, memory mapping
add_demo_target(demo_01_tlars
    demo_01_tlars.cpp
    T_ALGORITHM_LIBS
)

# Demo 02: T-LASSO basic usage, serialization, memory mapping
add_demo_target(demo_02_tlasso
    demo_02_tlasso.cpp
    T_ALGORITHM_LIBS
)

# Demo 03: T-Stepwise basic usage, serialization, memory mapping
add_demo_target(demo_03_tstepwise
    demo_03_tstepwise.cpp
    T_ALGORITHM_LIBS
)

# Demo 04: T-ENET basic usage, serialization, memory mapping
add_demo_target(demo_04_tenet
    demo_04_tenet.cpp
    T_ALGORITHM_LIBS
)

# Demo 05: T-OMP basic usage, serialization, memory mapping
add_demo_target(demo_05_tomp
    demo_05_tomp.cpp
    T_ALGORITHM_LIBS
)

# Demo 06: T-GP basic usage, serialization, memory mapping
add_demo_target(demo_06_tgp
    demo_06_tgp.cpp
    T_ALGORITHM_LIBS
)

# Demo 07: T-AC-GP basic usage, serialization, memory mapping
add_demo_target(demo_07_tacgp
    demo_07_tacgp.cpp
    T_ALGORITHM_LIBS
)


## Trex Selector based demos
# ===================================================================================

# Demo 08: TRex Selector basic usage, memory mapping based usage
set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex/trex_core)
add_demo_target(demo_08_trex
    demo_08_trex.cpp
    T_REX_SELECTOR_LIBS
)

# Demo 09: Screen TRex Selector basic usage, memory mapping based usage
#add_demo_target(demo_09_screen_trex
#   demo_09_screen_trex.cpp
#   T_REX_SELECTOR_LIBS
#)

# Demo 10: Biobank Screen TRex Selector basic usage, memory mapping based usage
#add_demo_target(demo_10_biobank_screen_trex
#   demo_10_biobank_screen_trex.cpp
#   T_REX_SELECTOR_LIBS
#)


# Matplot++ based demos
# ===================================================================================
# Demo 11: Plot dummy matrices
set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex/trex_core)
add_demo_target(demo_11_plot_dummy_matrices
    demo_11_plot_dummy_matrices.cpp
    T_REX_SELECTOR_LIBS
    Matplot++::matplot
)

set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex/trex_core)
add_demo_target(demo_12_biodata
    demo_12_biodata.cpp
    T_REX_SELECTOR_LIBS
    Matplot++::matplot
)


set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex/trex_core)
add_demo_target(demo_13_plot_fdr_tpr_results
    demo_13_plot_fdr_tpr_results.cpp
    T_REX_SELECTOR_LIBS
    Matplot++::matplot
)


# Demo 14: TRex Selector with HapGen Data
# ===================================================================================
# set(ADDITIONAL_INCLUDES ${CMAKE_SOURCE_DIR}/src/trex/trex_core)
# add_demo_target(demo_14_trex_hapgen
#     demo_14_trex_hapgen.cpp
#     T_REX_SELECTOR_LIBS
#     Matplot++::matplot
# )
