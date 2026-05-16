cmake_minimum_required(VERSION 3.24)

# Force Homebrew LLVM on macOS before project() if available
if(APPLE AND NOT DEFINED CMAKE_CXX_COMPILER)
    set(HOMEBREW_LLVM_PREFIX "/opt/homebrew/opt/llvm")
    if(EXISTS "${HOMEBREW_LLVM_PREFIX}/bin/clang++")
        set(CMAKE_C_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang" CACHE FILEPATH "C compiler")
        set(CMAKE_CXX_COMPILER "${HOMEBREW_LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "C++ compiler")
    endif()
endif()

project(trex_selector_python)

find_package(pybind11 CONFIG REQUIRED)

# Disable BLAS when wrapping in Python to avoid NumPy threading conflicts
set(TREX_USE_BLAS_LAPACK OFF CACHE BOOL "Disable BLAS for Python" FORCE)

# Disable Demos to avoid linker errors with executable targets
set(TREX_BUILD_DEMOS OFF CACHE BOOL "Disable Demos wrapping in Python" FORCE)

# Add the parent directory containing the main TRexSelector C++ project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/trex_core)

# Create the Python extension module mapped to _core
pybind11_add_module(_core src/trex_selector/bindings.cpp)

# Link against the underlying C++ libraries
target_link_libraries(_core PRIVATE
    trex::trex_selector_methods
    trex::utils
)

# Install the extension into the trex_selector package directory
install(TARGETS _core DESTINATION trex_selector)
