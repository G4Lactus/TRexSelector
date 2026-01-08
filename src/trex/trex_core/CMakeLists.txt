# src/trex/trex_core/CMakeLists.txt
# ===============================================================================
# CMake configuration for the trex_core module
# ===============================================================================


# ===============================================================================
# Define the trex_core library
# ===============================================================================
# Create a static library for the core trex functionalities
add_library(trex_core STATIC
    TRex_Selector.cpp
)

# Add alias for consistent naming
add_library(trex::core ALIAS trex_core)
# ===============================================================================


# ===============================================================================
# Include directores and link dependencies
# ===============================================================================
# Include directories for this library for hierarchical includes
target_include_directories(trex_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/>
)


# Link dependencies
target_link_libraries(trex_core PUBLIC
    # Internal dependencies
    trex::utils::memmap
    trex::utils::serialization
    trex::utils::openmp
    trex::utils::eval_metrics
    trex::utils::datagen
    # tsolver dependencies
    trex::tsolvers
    # ml_methods dependencies
    trex::ml_methods
)
# ===============================================================================


# ===============================================================================
# Compiler Settings
# ===============================================================================
target_compile_features(trex_core PUBLIC cxx_std_20)

# Set specific compile options
target_compile_options(trex_core PRIVATE
    -std=c++20
    -Wno-deprecated-declarations
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math -funroll-loops>
)
# ===============================================================================

# ----- End of File -------------------------------------------------------------
