# TRexSelector 2.0.1

A patch release. CRAN's daily checks of 2.0.0 reported "Installation failed" on the macOS builders for
R-release (x86_64) and R-oldrel (arm64 and x86_64), which compile with Apple clang 14.0.3. That compiler
accepts C++20 but its standard library has no ranges algorithms, and the backend used `std::ranges::find`
and `std::ranges::sort` at three places. The three calls are `std::find` and `std::sort` now, with identical
results.

On Linux with R 4.6 or R-devel the package compiled without OpenMP, because `R CMD config` no longer serves
`SHLIB_OPENMP_CXXFLAGS` there. The configure script reads the flags from R's Makeconf instead, so those builds
run their parallel regions again. The installation section of the README and of the getting-started vignette
points to CRAN first. Nothing else changes.

# TRexSelector 2.0.0

The package is rewritten as R bindings to the TRexSelector C++ library. The pure-R implementation of
1.0.0 and its dependencies (tlars, MASS, glmnet, boot, parallel, doParallel, foreach, doRNG) are gone.

## Breaking changes

* The functional interface of 1.0.0 (`trex()`, `FDP()`, `TPP()`, `add_dummies_GVS()` and the other
  helpers) is replaced by R6 classes and control lists:
  `TRexSelector$new(X, y, tFDR, control = trex_control(...))$select()`, with `trex_fdp()` and
  `trex_tpp()` as the metrics.
* R 4.3 or newer and a C++20 compiler are required. OpenMP is optional.

## New

* Selector variants as R6 classes: `TRexDASelector` (dependency-aware), `TRexGVSSelector` (grouped
  variable selection with the elastic net and the informed elastic net), `TRexScreeningSelector` and
  `TRexBiobankScreeningSelector` (screening), `TRexSPCASelector` (sparse PCA) and
  `TRexTikhonovSelector` (a general Tikhonov penalty).
* The twenty terminating solvers as classes of their own (`TLARS_Solver`, `TLASSO_Solver`, ...) with
  `tsolver_options()`, working on the caller's matrices without copying.
* Machine learning utilities: scalers, ridge and elastic-net regression with cross-validation, SVD,
  PCA and hierarchical clustering (the `ML*` classes and `ml_*` functions).
* Memory-mapped matrices (`mem_matrix()`, `mem_matrix_from()`), accepted everywhere a matrix is.
* Twelve vignettes, and `citation("TRexSelector")` lists the package entry with the method papers.
