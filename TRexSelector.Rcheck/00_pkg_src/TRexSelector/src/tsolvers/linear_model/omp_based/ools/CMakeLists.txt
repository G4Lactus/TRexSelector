cmake_minimum_required(VERSION 3.20)
project(ools
    VERSION   0.1.0
    LANGUAGES CXX
    DESCRIPTION "Orthogonal Least Squares Variable Selector (QR + Cholesky)"
)

# ── C++20 ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ── Eigen3 ──────────────────────────────────────────────────────
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# ── Library ─────────────────────────────────────────────────────
add_library(ools STATIC
    src/OOLS.cpp
)
target_include_directories(ools PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ools PUBLIC Eigen3::Eigen)
target_compile_options(ools PRIVATE
    -Wall -Wextra -Wpedantic -O2
)

# ── Demo executable ─────────────────────────────────────────────
add_executable(ools_demo demo/main.cpp)
target_link_libraries(ools_demo PRIVATE ools)
target_compile_options(ools_demo PRIVATE
    -Wall -Wextra -Wpedantic -O2
)
