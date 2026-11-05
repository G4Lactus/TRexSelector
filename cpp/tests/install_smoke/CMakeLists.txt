# ==================================================================================
# Install smoke consumer (install / find_package check).
#
# A minimal downstream project: it finds an INSTALLED TRexSelector through
# find_package and links one selector target. Driven by run_install_smoke.sh,
# which installs the SDK into a scratch prefix first; not part of the regular
# test build.
# ==================================================================================
cmake_minimum_required(VERSION 3.24)
project(trex_install_smoke CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TRexSelector REQUIRED)

add_executable(smoke smoke.cpp)
target_link_libraries(smoke PRIVATE TRexSelector::trex_selector_methods)
