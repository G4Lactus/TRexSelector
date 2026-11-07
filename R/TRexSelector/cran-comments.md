# TRexSelector 2.0.1

## Summary of the update

A patch release of 2.0.0, which is on CRAN since 2026-11-06. CRAN's daily checks of 2.0.0 report
"Installation failed" on r-release-macos-x86_64, r-oldrel-macos-arm64 and r-oldrel-macos-x86_64. Those
builders compile with Apple clang 14.0.3, whose standard library accepts `-std=gnu++20` but has no
ranges algorithms, and the compiled backend used `std::ranges::find` and `std::ranges::sort` at three
places. The three calls are `std::find` and `std::sort` now, with identical results.

The same check logs showed that the Linux builds compiled without OpenMP: on R 4.6 and R-devel,
`R CMD config SHLIB_OPENMP_CXXFLAGS` answers "no information for variable" although Makeconf defines
it, and the configure script relied on that query. It reads Makeconf directly now and writes
`$(SHLIB_OPENMP_CXXFLAGS)` into Makevars, the form Writing R Extensions asks for, as Makevars.win
already did. The compiler warnings in the package's own sources that the check logs showed (an unused
variable in the sequential build, three signed-unsigned comparisons in the R bindings) are gone. The
installation section of the README and of the getting-started vignette points to CRAN first. Nothing
else changes in the package's code or interface. The package has no reverse dependencies on CRAN.

## Test environments

* macOS 27.2 (Apple silicon), R 4.6.1: `R CMD check --as-cran`, also with the CPU-to-elapsed
  thresholds CRAN applies (`_R_CHECK_*_TIMING_CPU_TO_ELAPSED_THRESHOLD_` at 2.5).
* clang 14 with libc++ 14 (Ubuntu 24.04 container), the closest available match to the failing CRAN
  builders: the compiled backend builds without a warning, while the 2.0.0 sources fail there on the
  same `std::ranges` call.
* R-devel in a rocker/r-devel container (Debian, x86_64): `R CMD check --as-cran`.
* The package's GitHub Actions jobs: install, the testthat suite and `R CMD check` without vignettes
  on Ubuntu, macOS and Windows (R release), the sequential build without OpenMP on Ubuntu, and
  Windows with the daily R-devel snapshot.

## R CMD check results

0 errors | 0 warnings | 0 notes on macOS.

Two NOTEs in the R-devel container, both from the container itself:

* checking compilation flags used ... NOTE: non-portable flags `-Wdate-time`,
  `-Werror=format-security` and `-Wformat`. They come from the Makeconf of Debian's R-devel build,
  not from the package's configure or Makevars.
* checking HTML version of manual ... NOTE: math rendering skipped, as V8 is unavailable there.

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
