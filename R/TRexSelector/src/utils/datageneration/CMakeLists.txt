# src/utils/datageneration/CMakeLists.txt
# ==================================================================================
# CMake configuration for the datageneration utility module.
# ==================================================================================


# ==================================================================================
# 1. Append Header Files (.hpp)
# ==================================================================================
# utils_dummygen.hpp is public API (used by trex.hpp core).
# utils_datagen.hpp and utils_noisegen.hpp are build-tree only (demos + tests);
# they are NOT installed with the library.
target_sources(trex_utils PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/utils_dummygen.hpp
)
# ==================================================================================
