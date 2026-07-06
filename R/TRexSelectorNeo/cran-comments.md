# TRexSelector Package

## New submission

This is a new submission of the TRexSelector package (version 2.0.0). This version
is a complete architectural rework: the R bindings are now backed by a dedicated
C++20 library, replacing the previous dependency on the 'tlars' R package.

Changes from the previous CRAN version:

* C++20 backend replaces the 'tlars' dependency entirely.
* New R6-based API (TRexSelector, TRexDASelector, TRexGVSSelector,
   TRexScreeningSelector, TRexBiobankScreeningSelector).
* Dependency-aware, GVS, screening, and biobank screening variants added.
* Full testthat test suite added.
* Vignette added.

## R CMD check results

There were no ERRORs or WARNINGs.

One NOTE (both local and win-builder):

* "New submission" — expected for a first submission of this revised package.
* The GitHub repository (URL and BugReports fields) is currently private and
   therefore not yet reachable by the automated URL checker. It will be made
   public upon acceptance. The published paper DOI
   (https://doi.org/10.1016/j.sigpro.2025.109894) is reachable.

Compilation NOTE: three -Wunused warnings from RcppEigen/Eigen's own sparse
matrix headers appear during compilation. These originate in upstream Eigen
headers and are not from the package's own code.

Tested on:

* macOS aarch64 (Apple M-series, macOS Tahoe 26.5), R 4.5.3
* Windows: tested via CRAN win-builder service

## Downstream dependencies

There are currently no downstream dependencies for this package.
