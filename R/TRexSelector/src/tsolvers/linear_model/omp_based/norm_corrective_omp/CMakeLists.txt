cmake_minimum_required(VERSION 3.16)
project(NormCorrectiveGMP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(gmp_demo gmp_demo.cpp)
target_link_libraries(gmp_demo PRIVATE Eigen3::Eigen)
