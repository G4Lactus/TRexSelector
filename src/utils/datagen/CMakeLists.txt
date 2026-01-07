# src/utils/datagen/CMakeLists.txt
# ==============================================================================
# CMake configuration for the datagen utility module
# ==============================================================================


# ==============================================================================
# Library Target
# ==============================================================================
# Create an interface library for the data generation utilities
add_library(trex_utils_datagen INTERFACE)

# Add alias for consistent naming
add_library(trex::utils::datagen ALIAS trex_utils_datagen)


# ==============================================================================
# Include Directories for hierarchical includes (Generator Expressions)
# ==============================================================================
target_include_directories(trex_utils_datagen INTERFACE
    # Build-tree
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>      # Public API headers
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>          # Internal headers
    # Install-tree
    $<INSTALL_INTERFACE:include>                        # Installed headers
)


# ==============================================================================
# Link Dependencies
# ==============================================================================
target_link_libraries(trex_utils_datagen INTERFACE
    Eigen3::Eigen
    trex::utils::openmp
    trex::utils::memmap
)

# Compiler settings
target_compile_features(trex_utils_datagen INTERFACE cxx_std_20)
