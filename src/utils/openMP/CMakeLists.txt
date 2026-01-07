# src/utils/openMP/CMakeLists.txt
# ==================================================================================
# CMake configuration for the utils/openMP module
# ==================================================================================

# ==================================================================================
# Create an INTERFACE library (no compiled code, header-only)
# ==================================================================================
add_library(trex_utils_openmp INTERFACE)

# Add alias for consistent naming
add_library(trex::utils::openmp ALIAS trex_utils_openmp)

# ==================================================================================
# Inlcude Directories for hierarchical includes (Generator Expressions)
# ==================================================================================
target_include_directories(trex_utils_openmp INTERFACE
    # Build-tree
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>    # Public API headers
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>        # Internal headers
    # Install-tree
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:src>
)

# ==================================================================================
# Link Dependencies
# ==================================================================================
target_link_libraries(trex_utils_openmp INTERFACE
    OpenMP::OpenMP_CXX
)

# ==================================================================================
# Compile Features and Definitions
# ==================================================================================
target_compile_features(trex_utils_openmp INTERFACE cxx_std_20)
