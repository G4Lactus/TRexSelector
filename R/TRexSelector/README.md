<!-- README.md is generated from README.Rmd. Please edit that file -->

# TRexSelector

**Title**: The T-Rex selector for fast high-dimensional variable
selection with FDR control

**Description**: It performs fast variable selection in large-scale
high-dimensional settings while controlling the false discovery rate
(FDR) at a user-defined target level. Since version 2.0.0 the R package
provides bindings to the TRexSelector C++ library, which replaces the
pure-R implementation of the 1.x releases under the same package name.

**Papers**: The package is based on the papers

J. Machkour, M. Muma, and D. P. Palomar, “The terminating-random
experiments selector: Fast high-dimensional variable selection with
false discovery rate control,” *Signal Processing*, vol. 231, article
109894, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109894>)

J. Machkour, M. Muma, and D. P. Palomar, “High-dimensional false
discovery rate control for dependent variables,” *Signal Processing*,
vol. 234, article 109990, 2025.
(<https://doi.org/10.1016/j.sigpro.2025.109990>)

**Note**: The T-Rex selector performs terminated-random experiments
(T-Rex) using the T-LARS algorithm and fuses the selected active sets of
all random experiments to obtain a final set of selected variables. The
T-Rex selector provably controls the false discovery rate (FDR), i.e.,
the expected fraction of selected false positives among all selected
variables, at the user-defined target level while maximizing the number
of selected variables and, thereby, achieving a high true positive rate
(TPR) (i.e., power). The T-Rex selector can be applied in various
fields, such as genomics, financial engineering, or any other field that
requires a fast and FDR-controlling variable/feature selection method
for large-scale high-dimensional settings.

In the following sections, we show you how to install and use the
package.

## Installation

The package compiles its C++ backend on installation, so a C++20-capable
compiler is required (GCC ≥ 10, Clang ≥ 12, Apple Clang ≥ 14, or Rtools
on Windows), together with R ≥ 4.3.

OpenMP is optional. Without it the package compiles single-threaded and
`trex_get_max_threads()` returns 1. On Linux and Windows, R supplies the
OpenMP flags. CRAN’s R for macOS has none, so `configure` compiles with
the `omp.h` of Homebrew’s libomp (`brew install libomp`) and links R’s
own OpenMP runtime (`lib/libomp.dylib` of the R installation, the one
other packages load), or Homebrew’s runtime when R ships none. Setting
`TREX_OPENMP=no` forces the single-threaded build, for example
`TREX_OPENMP=no R CMD INSTALL R/TRexSelector`.

First install the R-side prerequisites (once, from an R session):

``` r
install.packages(c("Rcpp", "RcppEigen", "BH", "Rcereal", "R6"))
```

Then install ‘TRexSelector’ from
[GitHub](https://github.com/trexselector/TRexSelector) — the package
lives in the `R/TRexSelector/` subdirectory of the repository:

``` r
# Option 1: Install directly from GitHub
install.packages("remotes")
remotes::install_github("trexselector/TRexSelector", subdir = "R/TRexSelector")

# Option 2: Install from a local clone of the repository
# (from the shell, at the repository root)
# R CMD INSTALL R/TRexSelector
```

*(CRAN still carries the pure-R 1.x release. The submission of version
2.0.0 is in progress, and once it is accepted,
`install.packages("TRexSelector")` provides the C++-backed package.)*

You can open the help pages with:

``` r
library(TRexSelector)
help(package = "TRexSelector")
?TRexSelector
?trex_control
?trex_fdp
?trex_tpp
# etc.
```

To cite the package ‘TRexSelector’ in publications use:

``` r
citation("TRexSelector")
```

# Quick Start

This section illustrates the basic usage of the ‘TRexSelector’ package
to perform FDR-controlled variable selection in large-scale
high-dimensional settings based on the T-Rex selector.

1.  **First**, we generate a high-dimensional Gaussian data set with
    sparse support:

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

2.  **Second**, we perform FDR-controlled variable selection using the
    T-Rex selector for a target FDR of 5%. The selectors are R6 classes:
    construct with `$new(...)`, run with `$select()`, and read the
    results from active fields such as `$selected_indices` (1-based
    indices):

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

So, for a preset target FDR of 5%, the T-Rex selector has selected all
true active variables and there are no false positives in this example.

Note that users have to choose the target FDR according to the
requirements of their specific applications.

All selector variants (`TRexSelector`, `TRexDASelector`,
`TRexGVSSelector`, `TRexScreeningSelector`,
`TRexBiobankScreeningSelector`, `TRexSPCASelector`) follow the same
construct → `$select()` → inspect pattern.

## Dummy Strategies

The T-Rex selector calibrates the number of dummies in its L-loop.
`trex_control(lloop_strategy = )` says how the dummies of that loop are
produced, in every selector that takes the control.

| Strategy | Dummies | Storage |
|----|----|----|
| `"STANDARD"` (default) | fresh matrices at every L step | K matrices |
| `"HCONCAT"` | the same K matrices grow by p columns per L step | K matrices |
| `"SEEDED"` | the draws of `"STANDARD"`, re-derived from the seed at every step | none |
| `"PERMUTATION"` | permuted columns of the design | K matrices |
| `"PERMUTATION_SEEDED"` | the permutation re-derived from the seed | none |
| `"SKIPL"` | no L-loop, the cap `max_dummy_multiplier` is used directly | K matrices |

`"STANDARD"` draws fresh dummies at every step, the law of the T-Rex
paper and the default on every surface. The seeded strategies store
nothing and suit designs where K times L times p columns do not fit in
memory. `"SEEDED"` selects exactly as `"STANDARD"` for the same seed,
and `"PERMUTATION_SEEDED"` exactly as `"PERMUTATION"`. `"HCONCAT"` keeps
the grown law, which calibrates a larger L and gains 0.6 percentage
points of true positive rate (TPR) on average in a paired study of 500
trials, at the same false discovery rate (FDR). `"SKIPL"` fixes L at the
cap and skips the calibration, and under a large fixed L without the
stagnation stop its FDP estimate is anti-conservative (the selector
warns from `max_dummy_multiplier = 10` on), so the calibrated strategies
are the safe choice. The dummies are standard Normal by default, and
`trex_dummy_distribution()` offers the other laws. Vignette page 2,
`vignette("v02-trex-selector")`, carries the study and the seeds
section.

## Threads

The package runs serially by default: `trex_control()` has
`parallel_rnd_experiments = FALSE`, `max_outer_threads = 1` and
`max_inner_threads = 1`, and `tsolver_options()` has `n_threads = 1`.
The random experiments of a selector run one after the other, every
T-solver on one thread. Parallel work is opt-in, and one layer owns the
cores at a time.

- `trex_control(parallel_rnd_experiments = TRUE, max_outer_threads = m)`
  runs the K random experiments on `m` threads, at most K. Inside the
  parallel experiments every solver runs serially.
- `trex_control(max_inner_threads = m)` gives the solvers inside the
  experiments a budget of `m` threads for their passes over the design.
  The budget is live when the experiments run in sequence, which the
  lean L-loop strategies (`"SEEDED"`, `"PERMUTATION"`,
  `"PERMUTATION_SEEDED"`) always do. A pass is split only when it
  touches at least a million entries of the design, because the cost of
  forming a team of threads exceeds the gain below that. Two to four
  threads carry the gain, since the passes are memory-bandwidth bound.
  The seeded generators (the dummies, the synthetic designs) count their
  work in the same unit with the measured cost of a random draw (about
  80 entries) and of seeding an engine (up to 32,000), so they split
  from a few thousand entries.
- `tsolver_options(n_threads = m)` is the same budget for a T-solver
  used on its own, and `$set_num_threads()` changes it on a live solver.
- Every other kernel (the scalers, the clustering, the ML regressions,
  `trex_normalize_design()`, the dummy and data generators) runs on the
  library’s default budget, which starts at 1, so the package is
  single-threaded by default on every machine. `trex_set_num_threads(m)`
  sets that default for the session and returns the previous value,
  `trex_get_num_threads()` reads it, and the `n_threads` argument of the
  scalers, the clustering and the ML classes overrides it per object.
  Inside a selector these kernels take `max_inner_threads`.
  `trex_get_max_threads()` reports the threads OpenMP can run on the
  machine, an upper bound for a budget.

Every thread option is a whole number of at least 1; other values are
errors. A seeded run selects the same variables whatever the thread
settings. Without OpenMP every setting is a no-op and
`trex_get_max_threads()` returns 1 (see Installation). The parallelism
vignette has the details.

## Documentation

The package has twelve vignettes in [`vignettes/`](vignettes/) (built on
installation with `build_vignettes = TRUE`, listed by
`browseVignettes("TRexSelector")`): getting started, the T-Rex selector,
T-Rex+DA, T-Rex GVS, screening, T-Rex SPCA, T-Rex Tikhonov, the
T-solvers, the ML methods, memory mapping, working without data copies,
and parallelism. The help page of every selector, T-solver and machine
learning class ends with the papers it builds on, and
`citation("TRexSelector")` lists the package entry and the T-Rex method
papers.

## Links

T-Rex paper: <https://doi.org/10.48550/arXiv.2110.06048>

TRexSelector repository (C++ core, Python, and R packages):
[GitHub-TRexSelector](https://github.com/trexselector/TRexSelector)

Bug reports: [GitHub
Issues](https://github.com/trexselector/TRexSelector/issues)

## Acknowledgements

The package builds on the Eigen linear algebra library: G. Guennebaud,
B. Jacob et al., Eigen v3, <https://libeigen.gitlab.io/>, 2010. It binds
the C++ core through Rcpp and RcppEigen and relies on cereal (Rcereal)
for serialization and on Boost.Interprocess (BH) for memory mapping,
with OpenMP for the threads where the toolchain provides it.

## License

GPL ≥ 3 — see [LICENSE](../../LICENSE).

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version. It is distributed WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.
