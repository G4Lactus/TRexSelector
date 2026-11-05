# TRexSelector 2.0.0

## Summary of the update

This is a major update of TRexSelector (1.0.0 on CRAN since 2024-02-23). The package is rewritten as
R bindings to the TRexSelector C++ library. The pure-R implementation of 1.0.0 and its dependencies
(tlars, MASS, glmnet, boot, parallel, doParallel, foreach, doRNG) are gone and the compiled backend
takes their place; the package now links to Rcpp, RcppEigen, BH and Rcereal and imports R6.

The user interface changes with it. The functional interface of 1.0.0 (`trex()`, `FDP()`, `TPP()`,
`add_dummies_GVS()` and the other helpers) is replaced by R6 classes (`TRexSelector`,
`TRexDASelector`, `TRexGVSSelector`, `TRexScreeningSelector`, `TRexBiobankScreeningSelector`,
`TRexSPCASelector`, `TRexTikhonovSelector`) with control lists (`trex_control()` and its siblings)
and the metrics `trex_fdp()` and `trex_tpp()`. NEWS.md lists the changes. The package has no reverse
dependencies on CRAN.

## Test environments

* macOS 27.2 (Apple silicon), R 4.6.1: `R CMD check --as-cran`, also with the CPU-to-elapsed
  thresholds CRAN applies (`_R_CHECK_*_TIMING_CPU_TO_ELAPSED_THRESHOLD_` at 2.5).
* Ubuntu 24.04 (arm64 container), R 4.3.3, with and without OpenMP.
* The package's GitHub Actions jobs: install, the testthat suite and `R CMD check` without vignettes
  on Ubuntu, macOS and Windows (R release), and the sequential build without OpenMP on Ubuntu.
* win-builder (R-devel): pending, to be added before the upload.

## R CMD check results

0 errors | 0 warnings | 0 notes on macOS.

Two NOTEs on Ubuntu:

* checking installed package size ... NOTE: installed size is 8.3 MB, sub-directories of 1 MB or more:
  libs 5.7 MB. The compiled backend (the Eigen template instantiations of the solvers) accounts for
  it; the shared object is stripped on Linux (src/Makevars.in).
* checking DESCRIPTION meta-information ... NOTE: 'LinkingTo' for 'BH' is unused as it has no
  'include' directory. An artifact of Ubuntu's r-cran-bh binary, which ships no headers; the CRAN BH
  package has them, and the sources include the Boost headers through it.

Compilation emits `-Wunused-but-set-variable` warnings from RcppEigen's Eigen headers (SparseLU,
"variable 'count' set but not used" and similar). They originate in the upstream headers, not in the
package's code.

## System requirements and OpenMP

A C++20 compiler (GCC 10, Clang 12, Apple Clang 14 or newer, Rtools on Windows) and R 4.3 or newer.
OpenMP is optional: `configure` uses R's `SHLIB_OPENMP_CXXFLAGS` where R provides them (Linux,
Windows); on macOS without them it looks for an `omp.h` (Homebrew libomp or the mac.r-project.org
OpenMP package) and links R's bundled libomp, and without one it compiles sequentially. The
examples run in under 0.3 s each, and the vignettes limit themselves to two threads.

## Downstream dependencies

There are no reverse dependencies on CRAN.
