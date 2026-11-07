<!-- README.md is generated from README.Rmd. Please edit that file -->

# TRexSelector — R Package

**Fast variable selection in high-dimensional settings with control of
the false discovery rate (FDR).**

The T-Rex selector runs terminating random experiments with dummy
variables and fuses their selected active sets into the final set of
selected variables. It provably controls the false discovery rate (FDR),
the expected fraction of false positives among the selected variables,
at a user-defined target level while selecting as many variables as
possible, and so reaches a high true positive rate (TPR). It is fast and
memory-efficient enough for designs with millions of variables, as they
arise in genomics or financial engineering. Since version 2.0.0 the R
package provides bindings to the TRexSelector C++ library, which
replaces the pure-R implementation of the 1.x releases under the same
package name.

The method family is published in a series of papers. The two main
journal articles are

> J. Machkour, M. Muma, and D. P. Palomar, “The terminating-random
> experiments selector: Fast high-dimensional variable selection with
> false discovery rate control,” *Signal Processing*, vol. 231, article
> 109894, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109894>)
>
> J. Machkour, M. Muma, and D. P. Palomar, “High-dimensional false
> discovery rate control for dependent variables,” *Signal Processing*,
> vol. 234, article 109990, 2025.
> (<https://doi.org/10.1016/j.sigpro.2025.109990>)

The References section of the repository (see Links) lists the others,
and the help page of every class names the papers it builds on.

## Installation

The released package is on the Comprehensive R Archive Network (CRAN).
Install it from an R session:

``` r
install.packages("TRexSelector")
```

- CRAN provides binaries for Windows and macOS, so no compiler is needed
  there.
- On Linux, and for every install from source, the package compiles its
  C++ backend. This needs R ≥ 4.3 and a C++20 compiler (GCC ≥ 10, Clang
  ≥ 12, Apple Clang ≥ 14, or Rtools on Windows).
- OpenMP is optional and enables the threads described below. Where it
  is missing, the package runs on one thread and
  `trex_get_max_threads()` returns 1, which is also the check after an
  installation. The CRAN binary for Windows is built with OpenMP. On
  Linux, `install.packages()` compiles from source with the OpenMP flags
  of R, so the build has OpenMP. CRAN’s R for macOS has no OpenMP, so
  the macOS binary runs on one thread, while a build from source has
  OpenMP when Homebrew’s libomp is installed (the next block), because
  `configure` compiles with its `omp.h` and links the OpenMP runtime
  that R ships.
- Setting `TREX_OPENMP=no` forces the single-threaded build on every
  platform, for example `TREX_OPENMP=no R CMD INSTALL R/TRexSelector`.

For a threaded build on macOS, install libomp with Homebrew and then the
package from source:

``` sh
brew install libomp
```

``` r
install.packages("TRexSelector", type = "source")
```

The [GitHub repository](https://github.com/trexselector/TRexSelector)
holds the package in its `R/TRexSelector/` subdirectory. The `main`
branch carries the released sources, the version that is on CRAN, and
the development happens on the `dev_beyond` branch. An install from the
repository needs the R-side prerequisites (once, from an R session):

``` r
install.packages(c("Rcpp", "RcppEigen", "BH", "Rcereal", "R6"))
```

Then install the package from GitHub or from a local clone:

``` r
# Option 1: Install directly from GitHub (the main branch)
install.packages("remotes")
remotes::install_github("trexselector/TRexSelector", subdir = "R/TRexSelector")

# Option 2: Install from a local clone of the repository
# (from the shell, at the repository root)
# R CMD INSTALL R/TRexSelector
```

The help pages open with:

``` r
library(TRexSelector)
help(package = "TRexSelector")
?TRexSelector
?trex_control
?trex_fdp
?trex_tpp
# etc.
```

To cite the package in publications use:

``` r
citation("TRexSelector")
```

## Quick Start

This section illustrates the basic usage of the package on a small
high-dimensional example.

First, generate a Gaussian data set with sparse support:

``` r
library(TRexSelector)

# Setup
n <- 75 # number of observations
p <- 150 # number of variables
num_act <- 3 # number of true active variables
beta <- c(rep(1, times = num_act), rep(0, times = p - num_act)) # coefficient vector
true_actives <- which(beta > 0) # indices of true active variables

# Generate Gaussian data
set.seed(123)
X <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
y <- X %*% beta + stats::rnorm(n)
```

Second, run the T-Rex selector for a target FDR of 5%. The selectors are
R6 classes, so a selector is constructed with `$new(...)`, run with
`$select()`, and read through active fields such as `$selected_indices`
(1-based indices):

``` r
# Seed
set.seed(1234)

# Variable selection via T-Rex
sel <- TRexSelector$new(X, y, tFDR = 0.05, verbose = FALSE)
sel$select()

paste0("True active variables: ", paste(as.character(true_actives), collapse = ", "))
#> [1] "True active variables: 1, 2, 3"
paste0("Selected variables:    ", paste(as.character(sel$selected_indices), collapse = ", "))
#> [1] "Selected variables:    1, 2, 3"
```

For the preset target FDR of 5%, the T-Rex selector has selected all
true active variables and no false positive in this example. The target
FDR has to be chosen according to the requirements of the application.

Every selector of the next section follows the same pattern of
construction, `$select()` and inspection.

## What the package provides

### Selectors

Seven selectors cover the T-Rex method family. Each takes a control
list, which carries the T-solver choice, the dummy settings and the
thread budgets.

| Selector | Task | Control |
|----|----|----|
| `TRexSelector` | FDR-controlled variable selection, the T-Rex selector | `trex_control()` |
| `TRexDASelector` | dependency-aware (DA) dummies for correlated variables, T-Rex+DA | `trex_da_control()` |
| `TRexGVSSelector` | group variable selection (GVS), T-Rex GVS | `trex_gvs_control()` |
| `TRexScreeningSelector` | fast FDR-controlled screening of many variables, Screen-T-Rex | `trex_screen_control()` |
| `TRexBiobankScreeningSelector` | phenotype screening of genomic biobank designs | `trex_biobank_control()` |
| `TRexSPCASelector` | sparse principal component analysis (SPCA) under FDR control | `trex_spca_control()` |
| `TRexTikhonovSelector` | selection with a general Tikhonov penalty matrix | `trex_tikhonov_control()` |

The control list also sets the dummy strategy of the L-loop,
`lloop_strategy`. Its default `"STANDARD"` draws fresh dummies at every
step, as the T-Rex paper does, while the seeded strategies re-derive
them from the seed and store nothing, for designs whose dummy matrices
do not fit in memory. The dummies are standard Normal by default, and
`trex_dummy_distribution()` offers the other laws. The vignette of the
T-Rex selector, `vignette("v02-trex-selector")`, describes every
strategy and the seeds.

### T-solvers

Each random experiment runs a terminating solver, a T-solver, which
stops once a given number of dummies has entered its path. Twenty
T-solvers are available as R6 classes in four families,
`tsolver_options()` holds their options, and `trex_control(solver = )`
picks the one a selector uses.

- Least angle regression (LARS) based: `TLARS_Solver`, `TLASSO_Solver`,
  `TSTEPWISE_Solver`, `TSTAGEWISE_Solver`, `TENET_Solver`,
  `TENETAug_Solver`, `TIENET_Solver`, `TIENETAug_Solver` and
  `TIKNET_Solver`.
- Matching pursuit based: `TOMP_Solver`, `TGP_Solver`, `TACGP_Solver`,
  `TMP_Solver`, `TOOLS_Solver` and `TNCGMP_Solver`.
- Adaptive forward stepwise: `TAFS_Solver`.
- Coordinate descent (CD) based: `TCDLASSO_Solver`, `TCDENET_Solver`,
  `TCDIENET_Solver` and `TCDTIKNET_Solver`.

### ML methods

The machine learning (ML) building blocks the selectors share are
available on their own: the scalers `MLZScoreScaler` and
`MLLpNormScaler`, the decompositions `MLSVD` and `MLPCA`, the
regressions `MLRidge` and `MLElasticNet` with their cross-validated
variants `MLRidgeCV` and `MLElasticNetCV`, and agglomerative clustering
through `ml_agglomerative_cluster()` and `ml_cut_tree()`.

### Utilities

- Memory-mapped designs: `mem_matrix_from()` writes a matrix to a
  memory-mapped file and `mem_matrix()` wraps such a file, which the
  selectors and T-solvers read without a copy.
- Metrics: `trex_fdp()`, `trex_tpp()`, `trex_precision()` and
  `trex_recall()` on index sets, `trex_fdp_dense()` and
  `trex_tpp_dense()` on coefficient vectors.
- Design helpers: `trex_normalize_design()`,
  `trex_dummy_distribution()`, `trex_gamma_to_k()` and
  `trex_tikhonov_lambda2_power_rule()`.

### Threads

The package runs serially by default, and parallel work is opt-in.
`trex_control(parallel_rnd_experiments = TRUE, max_outer_threads = m)`
runs the random experiments of a selector on `m` threads,
`trex_control(max_inner_threads = m)` and
`tsolver_options(n_threads = m)` give the T-solvers a budget of `m`
threads for their passes over the design, and `trex_set_num_threads(m)`
sets the default budget of the scalers, the clustering and the ML
methods for the session, which `trex_get_num_threads()` reads.
`trex_get_max_threads()` reports the threads OpenMP can run on the
machine and returns 1 without OpenMP (see Installation). A seeded run
selects the same variables whatever the thread settings. The parallelism
vignette, `vignette("v12-parallelism")`, explains which layer owns the
cores.

## Documentation

The package has twelve vignettes, listed by
`browseVignettes("TRexSelector")` and opened as
`vignette("v02-trex-selector")`. An install from GitHub builds them with
`build_vignettes = TRUE`, and their sources are in the [vignettes
directory](https://github.com/trexselector/TRexSelector/tree/main/R/TRexSelector/vignettes)
of the repository.

1.  Getting started with TRexSelector
2.  The T-Rex selector
3.  T-Rex+DA: dependency-aware dummies
4.  T-Rex GVS: grouped variable selection
5.  Screening: Screen-T-Rex and the biobank screening
6.  T-Rex SPCA: sparse principal components under FDR control
7.  T-Rex Tikhonov: a general penalty matrix
8.  The T-solvers
9.  The ML methods
10. Memory mapping: disk-backed designs
11. Working without data copies
12. Parallelism: who owns the cores

The help page of every selector, T-solver and ML class ends with the
papers it builds on, and `citation("TRexSelector")` lists the package
entry and the T-Rex method papers.

## Links

- Project website: <https://trexselector.com>
- Repository of the C++ core and the R and Python packages:
  <https://github.com/trexselector/TRexSelector>
- Examples in R, Python and C++:
  <https://github.com/trexselector/TRexSelector_Examples>
- References of the method family:
  <https://github.com/trexselector/TRexSelector#references>
- Bug reports: <https://github.com/trexselector/TRexSelector/issues>

## Acknowledgements

The package builds on the Eigen linear algebra library: G. Guennebaud,
B. Jacob et al., Eigen v3, <https://libeigen.gitlab.io/>, 2010. It binds
the C++ core through Rcpp and RcppEigen and relies on cereal (Rcereal)
for serialization and on Boost.Interprocess (BH) for memory mapping,
with OpenMP for the threads where the toolchain provides it.

## License

GPL ≥ 3, see the [GNU General Public License
v3](https://www.gnu.org/licenses/gpl-3.0.html).

```text
This program is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
