# src/utils/datageneration/CMakeLists.txt
# ==================================================================================
# CMake configuration for the datageneration utility module.
# ==================================================================================


# ==================================================================================
# 1. Append Header Files (.hpp)
# ==================================================================================
# All three headers are public C++ API and are installed with the library.
# utils_dummygen.hpp  — dummy-matrix generation (used internally by TRexSelector).
# utils_datagen.hpp   — synthetic dataset generation for simulations/benchmarks.
# utils_noisegen.hpp  — noise distribution policies used by utils_datagen.hpp.
# R and Python bindings do not expose datagen/noisegen; C++ install only.
target_sources(trex_utils PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/utils_dummygen.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utils_datagen.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utils_noisegen.hpp
)
# ==================================================================================
