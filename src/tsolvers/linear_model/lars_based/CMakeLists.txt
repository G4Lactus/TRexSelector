# src/tsolvers/linear_model/lars_based/CMakeLists.txt
# ==============================================================================
# CMakeLists.txt for the LARS-based solvers for the linear regression model.
# ==============================================================================


# ==============================================================================
# 1. Append Source Files (.cpp - private)
# ==============================================================================
target_sources(trex_tsolvers PRIVATE
    # LARS based solvers
    # -----------------------------------------------
    ${CMAKE_CURRENT_LIST_DIR}/tenet_aug_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tenet_solver.cpp
    #${CMAKE_CURRENT_LIST_DIR}/tienet_solver.cpp
    #${CMAKE_CURRENT_LIST_DIR}/tienet_aug_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tlars_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tlasso_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tstagewise_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tstepwise_solver.cpp
)
# ==============================================================================


# ==============================================================================
# 2. Append Header Files (.hpp - public)
# ==============================================================================
target_sources(trex_tsolvers PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src          # project root /src
    FILES
        # LARS based solvers
        # -----------------------------------------------
        ${CMAKE_CURRENT_LIST_DIR}/tenet_aug_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tenet_solver.hpp
        #${CMAKE_CURRENT_LIST_DIR}/tienet_solver.hpp
        #${CMAKE_CURRENT_LIST_DIR}/tienet_aug_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tlars_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tlasso_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tstagewise_solver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/tstepwise_solver.hpp
)
# ==============================================================================
