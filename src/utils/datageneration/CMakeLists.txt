# src/utils/datageneration/CMakeLists.txt
# ==================================================================================
# CMake configuration for the datageneration utility module.
# ==================================================================================


# ==================================================================================
# 1. Append Header Files (.hpp - public)
# ==================================================================================
target_sources(trex_utils PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src      # project root src/
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/utils_datagen.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utils_dummygen.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utils_noisegen.hpp
)
# ==================================================================================
