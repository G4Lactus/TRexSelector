# trexselector — Python Package

[![PyPI version](https://badge.fury.io/py/trexselector.svg)](https://badge.fury.io/py/trexselector)

**Fast variable selection in high-dimensional settings with control of the false discovery rate (FDR).**

The T-Rex selector runs terminating random experiments with dummy variables and fuses their selected active sets into
the final set of selected variables. It provably controls the false discovery rate (FDR), the expected fraction of
false positives among the selected variables, at a user-defined target level while selecting as many variables as
possible, and so reaches a high true positive rate (TPR). It is fast and memory-efficient enough for designs with
millions of variables, as they arise in genomics or financial engineering. Since version 1.0.0 the Python package
provides bindings to the TRexSelector C++ library through pybind11, which replaces the pure-Python implementation of
the 0.6.x releases under the same package name. It is distributed on PyPI and imported as `trexselector`.

The method family is published in a series of papers. The two main journal articles are

> J. Machkour, M. Muma, and D. P. Palomar, "The terminating-random experiments selector: Fast high-dimensional variable selection with false discovery rate control," *Signal Processing*, vol. 231, article 109894, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109894>)
>
> J. Machkour, M. Muma, and D. P. Palomar, "High-dimensional false discovery rate control for dependent variables," *Signal Processing*, vol. 234, article 109990, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109990>)

The References section of the repository (see Links) lists the others, and the docstring of every class names the
papers it builds on.

## Installation

The released package is on the Python Package Index (PyPI). Install it into the active environment:

```bash
pip install trexselector
```

- PyPI provides binary wheels for Linux (x86_64), macOS (arm64) and Windows (x86_64) for Python 3.10 to 3.15, so no
  compiler is needed there. NumPy ≥ 1.26 is installed alongside.
- Every wheel bundles the OpenMP runtime of its platform, so the package runs on several threads out of the box.
  `trexselector.utils.get_max_threads()` reports the threads available, which is also the check after an
  installation.
- A build from source compiles the C++ backend. This needs a C++20 compiler, CMake ≥ 3.24 and the C++ dependencies
  Eigen3, Boost ≥ 1.80 and cereal, as the
  [C++ core README](https://github.com/trexselector/TRexSelector/blob/main/cpp/README.md) lists per platform. OpenMP
  is optional in that build and comes from the toolchain. On macOS the build picks Homebrew's LLVM when it is
  installed (`brew install llvm libomp`), since Apple clang has no OpenMP. Without OpenMP the package runs on one
  thread and `get_max_threads()` returns 1, and the CMake switch `TREX_WITH_OPENMP=OFF` forces that build.

The [GitHub repository](https://github.com/trexselector/TRexSelector) holds the package in its `Python/trexselector/`
subdirectory. The `main` branch carries the released sources, the version that is on PyPI, and the development
happens on the `dev_beyond` branch. A source install from the repository, from its `main` branch:

```bash
pip install git+https://github.com/trexselector/TRexSelector.git#subdirectory=Python/trexselector
```

The single-threaded build from a local clone, at the repository root:

```bash
pip install -C cmake.define.TREX_WITH_OPENMP=OFF Python/trexselector
```

The development install is an editable install without build isolation, so pybind11, scikit-build-core and CMake
must be available in the active environment:

```bash
pip install --no-build-isolation -e Python/trexselector

# Run the test suite against the installed package
pytest Python/trexselector/tests -q
```

## Quick Start

```python
import numpy as np
import trexselector

rng = np.random.default_rng(123)
n, p, num_act = 75, 150, 3        # observations, variables, true actives
beta = np.zeros(p)
beta[:num_act] = 1.0              # coefficient vector with sparse support
true_actives = np.nonzero(beta)[0].tolist()  # 0-based indices

X = rng.standard_normal((n, p))
y = X @ beta + rng.standard_normal(n)

sel = trexselector.TRexSelector(X, y, tFDR=0.05, verbose=False)
sel.select()

print(f"True active variables: {true_actives}")
print(f"Selected variables:    {sel.selected_indices}")
```

```text
True active variables: [0, 1, 2]
Selected variables:    [0, 1, 2]
```

For the preset target FDR of 5%, the T-Rex selector has selected all true active variables and no false positive in
this example. The target FDR (`tFDR`) has to be chosen according to the requirements of the application.

Every selector of the next section follows the same pattern of construction, `select()` and inspection.

## What the package provides

### Selectors

Seven selectors cover the T-Rex method family, all at the package root. Each takes a control object, which carries
the T-solver choice, the dummy settings and the thread budgets, and returns a result object from `select()`.

| Selector | Task | Control |
| --- | --- | --- |
| `TRexSelector` | FDR-controlled variable selection, the T-Rex selector | `TRexControlParameter` |
| `TRexDASelector` | dependency-aware (DA) dummies for correlated variables, T-Rex+DA | `TRexDAControlParameter` |
| `TRexGVSSelector` | group variable selection (GVS), T-Rex GVS | `TRexGVSControlParameter` |
| `TRexScreeningSelector` | FDR-controlled screening of many variables, Screen-T-Rex | `ScreenTRexControlParameter` |
| `TRexBiobankScreeningSelector` | phenotype screening of genomic biobank designs | `BiobankScreenTRexControl` |
| `TRexSPCASelector` | sparse principal component analysis (SPCA) under FDR control | `TRexSPCAControlParameter` |
| `TRexTikhonovSelector` | selection with a general Tikhonov penalty matrix | `TRexTikhonovControlParameter` |

The control object also sets the dummy strategy of the L-loop, `lloop_strategy`, with the `LLoopStrategy` enum. Its
default `STANDARD` draws fresh dummies at every step, as the T-Rex paper does, while the seeded strategies re-derive
them from the seed and store nothing, for designs whose dummy matrices do not fit in memory. The dummies are standard
Normal by default, and `DummyDistribution` offers the other laws.

### T-solvers

Each random experiment runs a terminating solver, a T-solver, which stops once a given number of dummies has entered
its path. Twenty T-solvers are available as classes of their own under `trexselector.tsolvers.ad`, in four families
that are its submodules. `SolverOptions` holds their options, and `TRexControlParameter.solver_type` picks the one a
selector uses, with the `SolverTypeForTRex` enum.

- `lars_based`, least angle regression (LARS) based: `TLARS_Solver`, `TLASSO_Solver`, `TSTEPWISE_Solver`,
  `TSTAGEWISE_Solver`, `TENET_Solver`, `TENETAug_Solver`, `TIENET_Solver`, `TIENETAug_Solver` and `TIKNET_Solver`.
- `omp_based`, matching pursuit based: `TOMP_Solver`, `TGP_Solver`, `TACGP_Solver`, `TMP_Solver`, `TOOLS_Solver` and
  `TNCGMP_Solver`.
- `afs_based`, adaptive forward stepwise: `TAFS_Solver`.
- `cd_based`, coordinate descent (CD) based: `TCDLASSO_Solver`, `TCDENET_Solver`, `TCDIENET_Solver` and
  `TCDTIKNET_Solver`.

### ML methods

The machine learning (ML) building blocks the selectors share are available on their own under
`trexselector.ml_methods`: the scalers `ZScoreScaler` and `LpNormScaler`, the decompositions `SVDSolver` and `PCA`,
the regressions `RidgeSolver` and `ElasticNet` with their cross-validated variants `RidgeCV` and `ElasticNetCV`, and
agglomerative clustering through `agglomerative_cluster()` and `cut_tree()` of `trexselector.ml_methods.clustering`.

### Utilities

- Memory-mapped designs: `trexselector.utils.convert_to_memory_mapped()` writes an array to a memory-mapped file and
  `MemoryMappedMatrix` opens such a file, which the selectors and T-solvers read without a copy.
- Metrics in `trexselector.utils`: `compute_fdp()`, `compute_tpp()`, `compute_precision()` and `compute_recall()`
  on index sets, `compute_fdp_dense()` and `compute_tpp_dense()` on coefficient vectors, and the counts
  `true_positives_count()`, `false_positives_count()` and `false_negatives_count()`.
- Design helpers at the package root: `normalize_design()`, `gamma_to_k()`, `build_group_centering_k()` and
  `tikhonov_lambda2_power_rule()`.

### Threads

The package runs serially by default, and parallel work is opt-in. `TRexControlParameter(parallel_rnd_experiments=True,
max_outer_threads=m)` runs the random experiments of a selector on `m` threads, `max_inner_threads=m` and
`SolverOptions(n_threads=m)` give the T-solvers a budget of `m` threads for their passes over the design, and
`trexselector.utils.set_num_threads(m)` sets the default budget of the scalers, the clustering and the ML methods for
the process, which `get_num_threads()` reads. `get_max_threads()` reports the threads OpenMP can run on the machine
and returns 1 without OpenMP (see Installation). Every thread option is a whole number of at least 1, and other values
raise `ValueError`. A seeded run selects the same variables whatever the thread settings.

## Documentation

The docstring of every class and function is the reference, reachable with `help(trexselector.TRexSelector)` and the
other names, and the stub files of the package give the signatures to an editor. The
[Examples repository](https://github.com/trexselector/TRexSelector_Examples) carries worked demos in Python, R and
C++, and the R package's twelve vignettes describe the methods in prose with the same parameters.

## Links

- Project website: <https://trexselector.com>
- Repository of the C++ core and the R and Python packages: <https://github.com/trexselector/TRexSelector>
- Examples in R, Python and C++: <https://github.com/trexselector/TRexSelector_Examples>
- References of the method family: <https://github.com/trexselector/TRexSelector#references>
- Citation metadata: [CITATION.cff](https://github.com/trexselector/TRexSelector/blob/main/CITATION.cff) at the
  repository root
- Bug reports: <https://github.com/trexselector/TRexSelector/issues>

## Acknowledgements

The library builds on the Eigen linear algebra library: G. Guennebaud, B. Jacob et al., Eigen v3,
<https://libeigen.gitlab.io/>, 2010. It binds the C++ core through pybind11 and relies on cereal for
serialization, on Boost.Interprocess for memory mapping and on OpenMP for the threads, with an optional
BLAS/LAPACK backend.

## License

GPL ≥ 3, see [LICENSE](https://github.com/trexselector/TRexSelector/blob/main/Python/trexselector/LICENSE) or the
[GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.html).

```text
This program is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
