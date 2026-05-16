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
        ${PROJECT_SOURCE_DIR}/include     # Public headers
        ${PROJECT_SOURCE_DIR}/src         # Source headers
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
# Demo Targets are now distributed to subfolder CMakeLists.txt files
# ===================================================================================

# Add subdirectories for each demo group
add_subdirectory(ml_methods)
add_subdirectory(tsolvers)
add_subdirectory(trex_selector_methods)
add_subdirectory(applications)
#
