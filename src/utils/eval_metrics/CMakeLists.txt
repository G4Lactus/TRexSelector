# src/utils/eval_metrics/CMakeLists.txt
# ==============================================================================
# CMake configuration for the eval_metrics utility module
# ==============================================================================


# ==============================================================================
# Library Target
# ==============================================================================
# Create an interface library for the evaluation metrics utilities
add_library(trex_utils_eval_metrics INTERFACE)

# Add alias for consistent naming
add_library(trex::utils::eval_metrics ALIAS trex_utils_eval_metrics)


# ==============================================================================
# Include Directories for hierarchical includes (Generator Expressions)
# ==============================================================================
target_include_directories(trex_utils_eval_metrics INTERFACE
    # Build-tree
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>      # Public API headers
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>          # Internal headers
    # Install-tree
    $<INSTALL_INTERFACE:include>                        # Installed headers
)


# ==============================================================================
# Link Dependencies
# ==============================================================================
target_link_libraries(trex_utils_eval_metrics INTERFACE
    Eigen3::Eigen
)

# Compiler settings
target_compile_features(trex_utils_eval_metrics INTERFACE cxx_std_20)
