#!/bin/bash
set -e
# Always run relative to this script's location (R/TRexSelector/), regardless
# of where it is invoked from — prevents accidentally targeting the repo root src/.
cd "$(dirname "$0")"

echo "=== Preparing TRexSelector for CRAN ==="

echo "Cleaning existing bundled C++ code..."
rm -rf src/ml_methods
rm -rf src/trex_selector_methods
rm -rf src/tsolvers
rm -rf src/utils

echo "Copying C++ backend files from root src/..."
cp -R ../../src/ml_methods src/
cp -R ../../src/trex_selector_methods src/
cp -R ../../src/tsolvers src/
cp -R ../../src/utils src/

echo "Stripping CMake artifacts and unused code..."
find src -type f -name "CMakeLists.txt" -delete
rm -rf src/tsolvers/linear_model/omp_based/ools
find src -type f -name "*demo*.cpp" -delete
find src -type f -name "main.cpp" -delete

echo "Regenerating Makevars files with fresh OBJECTS list..."
cd src
CPP_SOURCES=$(find . -type f -name "*.cpp" | sed 's|^\./||' | sort | tr '\n' ' ')
CPP_OBJECTS=$(echo "$CPP_SOURCES" | sed 's/\.cpp/.o/g')
cd ..

cat << MAKEVARS > src/Makevars
CXX_STD = CXX20
PKG_CPPFLAGS = -I. -DCRAN_BUILD=1 -DNDEBUG
PKG_CXXFLAGS = -O3 -funroll-loops -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
OBJECTS = $CPP_OBJECTS
PKG_LIBS = \$(LAPACK_LIBS) \$(BLAS_LIBS) \$(FLIBS) /opt/homebrew/opt/libomp/lib/libomp.dylib
MAKEVARS

cat << MAKEVARS > src/Makevars.in
CXX_STD = CXX20
PKG_CPPFLAGS = -I. -DCRAN_BUILD=1 -DNDEBUG
PKG_CXXFLAGS = -O3 -funroll-loops @OPENMP_CXXFLAGS@
OBJECTS = $CPP_OBJECTS
PKG_LIBS = \$(LAPACK_LIBS) \$(BLAS_LIBS) \$(FLIBS) @OPENMP_LIBS@
MAKEVARS

cat << MAKEVARS > src/Makevars.win
CXX_STD = CXX20
PKG_CPPFLAGS = -I. -DCRAN_BUILD=1 -DNDEBUG
PKG_CXXFLAGS = -O3 -funroll-loops \$(SHLIB_OPENMP_CXXFLAGS)
OBJECTS = $CPP_OBJECTS
PKG_LIBS = \$(LAPACK_LIBS) \$(BLAS_LIBS) \$(FLIBS) \$(SHLIB_OPENMP_CXXFLAGS)
MAKEVARS

echo "Ready for R CMD build."
