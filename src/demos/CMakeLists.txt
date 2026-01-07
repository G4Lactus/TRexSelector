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
# Common libraries for All demos
# ===================================================================================
set(DEMO_LIBS
    # TRex library components: bring in includes of dependencies
    trex::utils::datagen            # Data generation utilities
    trex::ml_methods                # ML tools
    trex::tsolvers                  # T-Algorithm solvers

    # Core dependencies not covered above
    Boost::headers
    Boost::iostreams
    # Linear algebra backends
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)


# ==================================================================================
# Macro: add_demo_target
# ==================================================================================
# Creates a demo executable with proper include paths, linking, and compile flags.
#
# Usage:
#   add_demo_target(<target_name> <demo_file> "<solver_sources>" [additional_libraries...])
#
# Arguments:
#   target_name          - Name of the demo executable (e.g., demo_00_scaler)
#   demo_file            - Source file for the demo (e.g., demo_00_scaler.cpp)
#   solver_sources       - Semicolon-separated list of solver .cpp files (can be empty)
#   additional_libraries - (Optional) Additional libraries to link against
#
# Example:
#   add_demo_target(demo_01_tlars demo_01_tlars.cpp "${TLARS_SOURCES}")
#
# ==================================================================================
macro(add_demo_target target_name demo_file)
    # Create executable
    add_executable(${target_name}
        ${demo_file}
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include     # Public headers
        ${CMAKE_SOURCE_DIR}/src         # Source headers
    )

    # Link libraries
    target_link_libraries(${target_name} PRIVATE
        ${DEMO_LIBS}
        ${ARGN}         # Additional libraries passed to the macro
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
endmacro()
# ------ End of macro ------


# ===================================================================================
# Demo Targets
# ===================================================================================
# Template for new demos:
# add_demo_target(<target_name>
#   <demo_file>
#   "<solver_sources>"
#   [additional_libraries...]
# )
# ==================================================================================

## T-Algorithm based demos
# ==================================================================================
# Demo 00: Data preprocessing (StandardScaler, Normalizer)
add_demo_target(demo_00_scaler
    demo_00_scaler.cpp
)

# Demo 01: T-LARS basic usage, serialization, memory mapping
add_demo_target(demo_01_tlars
    demo_01_tlars.cpp
)

# Demo 02: T-LASSO basic usage, serialization, memory mapping
add_demo_target(demo_02_tlasso
    demo_02_tlasso.cpp
)

# Demo 03: T-Stepwise basic usage, serialization, memory mapping
add_demo_target(demo_03_tstepwise
    demo_03_tstepwise.cpp
)

# Demo 04: T-ENET basic usage, serialization, memory mapping
add_demo_target(demo_04_tenet
    demo_04_tenet.cpp
)

# Demo 05: T-OMP basic usage, serialization, memory mapping
add_demo_target(demo_05_tomp
    demo_05_tomp.cpp
)

# Demo 06: T-GP basic usage, serialization, memory mapping
add_demo_target(demo_06_tgp
    demo_06_tgp.cpp
)

# Demo 07: T-AC-GP basic usage, serialization, memory mapping
add_demo_target(demo_07_tacgp
    demo_07_tacgp.cpp
)


## Trex Selector based demos
# ===================================================================================

# Demo 08: TRex Selector basic usage, memory mapping based usage
#add_demo_target(demo_08_trex demo_08_trex.cpp "${TREX_SOURCES}")

# Demo 09: Screen TRex Selector basic usage, memory mapping based usage
#add_demo_target(demo_09_screen_trex demo_09_screen_trex.cpp "${SCREEN_TREX_SOURCES}")

# Demo 10: Biobank Screen TRex Selector basic usage, memory mapping based usage
#add_demo_target(demo_10_biobank_screen_trex demo_10_biobank_screen_trex.cpp "${BIOSCREEN_TREX_SOURCES}")
