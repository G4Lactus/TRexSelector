# src/tsolvers/ad/linear_model/CMakeLists.txt
# ==============================================================================
# CMakeLists.txt for the linear regression model.
# ==============================================================================

# ==============================================================================
# 1. Group Headers (.hpp - public)
# ==============================================================================
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src          # project root /src
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/afs_based.hpp
        ${CMAKE_CURRENT_LIST_DIR}/cd_based.hpp
        ${CMAKE_CURRENT_LIST_DIR}/lars_based.hpp
        ${CMAKE_CURRENT_LIST_DIR}/omp_based.hpp
)
# ==============================================================================


# ==============================================================================
# 2. Append Source Files
# ==============================================================================
add_subdirectory(cd_based)
add_subdirectory(lars_based)
add_subdirectory(omp_based)
add_subdirectory(afs_based)
# ==============================================================================
