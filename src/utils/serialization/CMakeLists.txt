# src/utils/serialization/CMakeLists.txt
# ==============================================================================
# CMake configuration for Cereal serialization adapters (headers only)
# ==============================================================================

# ==============================================================================
# Library Target
# ==============================================================================
# Create an INTERFACE library for serialization utilities
add_library(trex_utils_serialization INTERFACE)

# Add alias for consistent naming
add_library(trex::utils::serialization ALIAS trex_utils_serialization)


# ==============================================================================
# Include Directories for hierarchical includes (Generator Expressions)
# ==============================================================================
target_include_directories(trex_utils_serialization INTERFACE
    # Build-tree
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>      # Public API headers
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>          # Internal headers
    # Install-tree
    $<INSTALL_INTERFACE:include>                        # Installed headers
)


# ==============================================================================
# Link dependencies (header-only, but depends on other libraries)
# ==============================================================================
target_link_libraries(trex_utils_serialization INTERFACE
    Eigen3::Eigen
    # if armadillo gets added -> armadillo::arma
)

# Compiler settings
target_compile_features(trex_utils_serialization INTERFACE cxx_std_20)
