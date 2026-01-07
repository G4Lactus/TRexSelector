# src/utils/memmap/CMakeLists.txt
# ==============================================================================
# CMake configuration for the memmap utility module
# ==============================================================================

# Create a static library for the memory-mapped matrix
add_library(trex_utils_memmap STATIC
    MemoryMappedMatrix.cpp
)

# Include directories for this library for hierarchical includes
target_include_directories(trex_utils_memmap PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(trex_utils_memmap PUBLIC
    Eigen3::Eigen
    Boost::headers
    Boost::iostreams
)

# Compiler settings
# Set c++20 requirement
target_compile_features(trex_utils_memmap PUBLIC cxx_std_20)

# Set specific compile options
target_compile_options(trex_utils_memmap PRIVATE
    -std=c++20
    -Wno-deprecated-declarations
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math -funroll-loops>
)

# Add alias for consistent naming
add_library(trex::utils::memmap ALIAS trex_utils_memmap)
