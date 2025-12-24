# src/CMakeLists.txt
set(DIRSRC ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(SUBDIR "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

# ------ Common source files ------
set(COMMON_SOURCES
    ${DIRSRC}/Normalizer.cpp
    ${DIRSRC}/StandardScaler.cpp
)

# ----- Solver sources ------
set(TLARS_SOURCES ${DIRSRC}/TLARS_Solver.cpp)
set(TLASSO_SOURCES ${DIRSRC}/TLASSO_Solver.cpp ${TLARS_SOURCES})
set(TSTEPWISE_SOURCES ${DIRSRC}/TSTEPWISE_Solver.cpp ${TLARS_SOURCES})
set(TENET_SOURCES ${DIRSRC}/TENET_Solver.cpp ${TLARS_SOURCES})
set(TOMP_SOURCES ${DIRSRC}/TOMP_Solver.cpp)
set(TGP_SOURCES ${DIRSRC}/TGP_Solver.cpp ${TOMP_SOURCES})
set(TACGP_SOURCES ${DIRSRC}/TACGP_Solver.cpp ${TGP_SOURCES} ${TOMP_SOURCES})
set(TREX_SOURCES
    ${DIRSRC}/TRex_Selector.cpp
    ${TLARS_SOURCES}
    ${TLASSO_SOURCES}
    ${TSTEPWISE_SOURCES}
    ${TENET_SOURCES}
    ${TOMP_SOURCES}
    ${TGP_SOURCES}
    ${TACGP_SOURCES}
)


# ----- Common libraries -----
set(COMMON_LIBS
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    cereal::cereal
    ${BLAS_LIBRARIES}    # From find_package(BLAS)
    ${LAPACK_LIBRARIES}  # From find_package(LAPACK)
)

# ----- Macro to configure demo target -----
macro(add_demo_target target_name demo_file solver_sources)
    add_executable(${target_name}
        ${DIRSRC}/${demo_file}
        ${solver_sources}
        ${COMMON_SOURCES}
    )

    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(${target_name} PRIVATE ${COMMON_LIBS})

    # *** CRITICAL: Tell Eigen to use BLAS ***
    target_compile_definitions(${target_name} PRIVATE
        EIGEN_USE_BLAS          # Enable BLAS backend
    )

    target_compile_options(${target_name} PRIVATE
        -std=c++20
        -Wno-deprecated-declarations
        $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -ffast-math -funroll-loops>
    )

    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debug/${SUBDIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/release/${SUBDIR}"
    )
endmacro()


# ----- Demo targets -----
add_demo_target(demo_00_scaler demo_00_scaler.cpp "")
add_demo_target(demo_01_tlars demo_01_tlars.cpp "${TLARS_SOURCES}")
add_demo_target(demo_02_tlasso demo_02_tlasso.cpp "${TLASSO_SOURCES}")
add_demo_target(demo_03_tstepwise demo_03_tstepwise.cpp "${TSTEPWISE_SOURCES}")
add_demo_target(demo_04_tenet demo_04_tenet.cpp "${TENET_SOURCES}")
add_demo_target(demo_05_tomp demo_05_tomp.cpp "${TOMP_SOURCES}")
add_demo_target(demo_06_tgp demo_06_tgp.cpp "${TGP_SOURCES}")
add_demo_target(demo_07_tacgp demo_07_tacgp.cpp "${TACGP_SOURCES}")
add_demo_target(demo_08_trex demo_08_trex.cpp "${TREX_SOURCES}")


# add_demo_target(demo_XX_backend demo_XX_backend.cpp "")
