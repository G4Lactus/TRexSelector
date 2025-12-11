# src/CMakeLists.txt
# =============================
# Builds all executables from source files
# Each target outputs to build/ debug/ or release/

set(DIRSRC ${CMAKE_CURRENT_SOURCE_DIR})


find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
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

# ----- Common libraries -----
set(COMMON_LIBS
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    /opt/homebrew/opt/openblas/lib/libopenblas.dylib
    /opt/homebrew/opt/lapack/lib/liblapack.dylib
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
        ${EIGEN3_INCLUDE_DIR}
    )

    target_link_libraries(${target_name} PRIVATE ${COMMON_LIBS})

    target_compile_options(${target_name} PRIVATE
        -std=c++20
        -Wno-deprecated-declarations
        $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>
    )

    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debug/${SUBDIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/release/${SUBDIR}"
    )
endmacro()


# ----- Demo targets -----

# --- demo_00_scaler ---
add_demo_target(demo_00_scaler demo_00_scaler.cpp "")

## --- demo_01_tlars ---
add_demo_target(demo_01_tlars demo_01_tlars.cpp "${TLARS_SOURCES}")

## --- demo_02_tlasso ---
#add_demo_target(demo_02_tlasso demo_02_tlasso.cpp "${TLASSO_SOURCES}")
#
## --- demo_03_tstepwise ---
#add_demo_target(demo_03_tstepwise demo_03_tstepwise.cpp "${TSTEPWISE_SOURCES}")
#
## --- demo_04_tenet ---
#add_demo_target(demo_04_tenet demo_04_tenet.cpp "${TENET_SOURCES}")
#
## --- demo_05_tomp ---
#add_demo_target(demo_05_tomp demo_05_tomp.cpp "${TOMP_SOURCES}")
#
## --- demo_06_tgp ---
#add_demo_target(demo_06_tgp demo_06_tgp.cpp "${TGP_SOURCES}")
#
## --- demo_07_tacgp ---
#add_demo_target(demo_07_tacgp demo_07_tacgp.cpp "${TACGP_SOURCES}")
