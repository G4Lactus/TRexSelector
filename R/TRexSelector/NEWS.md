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
