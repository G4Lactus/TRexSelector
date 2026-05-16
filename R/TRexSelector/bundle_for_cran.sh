#!/bin/bash
set -e
echo "=== Preparing TRexSelector for CRAN ==="

echo "Cleaning existing bundled code..."
rm -rf src/ml_methods
rm -rf src/trex_selector_methods
rm -rf src/tsolvers
rm -rf src/utils
rm -rf src/trex_core

echo "Copying C++ backend files..."
cp -R ../../src/ml_methods src/
cp -R ../../src/trex_selector_methods src/
cp -R ../../src/tsolvers src/
cp -R ../../src/utils src/

echo "Cleaning out unused/demo C++ code that breaks compilation..."
rm -rf src/tsolvers/linear_model/omp_based/ools
find src -type f -name "*demo*.cpp" -delete
find src -type f -name "main.cpp" -delete

echo "Creating CRAN-friendly Makevars.in..."
cd src
CPP_SOURCES=$(find . -type f -name "*.cpp" | sed 's|^\./||' | tr '\n' ' ')
CPP_OBJECTS=$(echo "$CPP_SOURCES" | sed 's/\.cpp/.o/g')
cd ..

cat << MAKEVARS > src/Makevars.in
CXX_STD = CXX20
PKG_CPPFLAGS = -I. -DCRAN_BUILD=1 -I./tsolvers/linear_model/omp_based @OPENMP_CXXFLAGS@
OBJECTS = $CPP_OBJECTS
PKG_LIBS = \$(LAPACK_LIBS) \$(BLAS_LIBS) \$(FLIBS) @OPENMP_LIBS@
MAKEVARS

cat << MAKEVARS > src/Makevars.win
CXX_STD = CXX20
PKG_CPPFLAGS = -I. -DCRAN_BUILD=1 -I./tsolvers/linear_model/omp_based \$(SHLIB_OPENMP_CXXFLAGS)
OBJECTS = $CPP_OBJECTS
PKG_LIBS = \$(LAPACK_LIBS) \$(BLAS_LIBS) \$(FLIBS) \$(SHLIB_OPENMP_CXXFLAGS)
MAKEVARS

echo "Ready for R CMD build."
