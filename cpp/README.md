# TRexSelector — C++ Core Library

**Fast variable selection in high-dimensional settings with control of the false discovery rate (FDR).**

The C++20 core of the T-Rex selector: every selector variant, terminating solver and supporting
kernel is implemented here once and reaches R and Python through thin binding layers. The T-Rex
selector runs terminating random experiments with dummy variables and fuses their selected active
sets into the final set of selected variables. It provably controls the false discovery rate (FDR),
the expected fraction of false positives among the selected variables, at a user-defined target level
while selecting as many variables as possible, and so reaches a high true positive rate (TPR). This
README covers the dependencies, building the library and its tests, installing and consuming it from
a C++ project, and what the library provides. The [repository README](../README.md) tells the story
of the method.

The method family is published in a series of papers. The two main journal articles are

> J. Machkour, M. Muma, and D. P. Palomar, "The terminating-random experiments selector: Fast high-dimensional variable selection with false discovery rate control," *Signal Processing*, vol. 231, article 109894, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109894>)
>
> J. Machkour, M. Muma, and D. P. Palomar, "High-dimensional false discovery rate control for dependent variables," *Signal Processing*, vol. 234, article 109990, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109990>)

The [References section](../README.md#references) of the repository README lists the others, and the
documentation comment of every class names the papers it builds on.

---

## Dependencies & Platform Compatibility

| Library                                        | Role                                      | Version    | Required |
|------------------------------------------------|-------------------------------------------|------------|----------|
| [Eigen3](https://libeigen.gitlab.io/)          | Linear algebra                            | any recent | Yes      |
| [Boost](https://www.boost.org)                 | Headers + iostreams (memory-mapped files) | ≥ 1.80     | Yes      |
| [Cereal](https://uscilab.github.io/cereal)     | Serialization of solver state             | any recent | Yes      |
| [OpenMP](https://www.openmp.org)               | Parallelisation of random experiments     | ≥ 3.1      | Optional |
| BLAS / LAPACK                                  | Accelerated linear algebra backend        | —          | Optional |

*Without OpenMP the library builds single-threaded. CMake then reports "OpenMP not found:
building single-threaded", and `omp_get_max_threads()` returns 1 through the stubs in
`src/utils/openmp/utils_openmp.hpp`. `-DTREX_WITH_OPENMP=OFF` forces that build even where
OpenMP exists.*

**Platform Support:**

* **macOS (arm64):** Fully tested (C++ core, R, and Python packages). BLAS/LAPACK defaults to Apple Accelerate.
* **Linux (x86_64):** Supported (OpenBLAS is used).
* **Windows (x86_64):** Supported.

*The BLAS/LAPACK backend can be disabled at configure time with `-DTREX_USE_BLAS_LAPACK=OFF`.*

The library builds on the Eigen linear algebra library: G. Guennebaud, B. Jacob et al., Eigen v3,
<https://libeigen.gitlab.io/>, 2010. It also relies on cereal for serialization, on Boost.Interprocess for
memory mapping and on OpenMP for the threads, with an optional BLAS/LAPACK backend. The R package binds the
library through Rcpp and RcppEigen, the Python package through pybind11.

---

## Installing the Dependencies

**macOS (Homebrew):**

```bash
brew install cmake llvm eigen cereal boost openblas libomp
```

*(CMake prefers Homebrew LLVM and its libomp for OpenMP. A compiler without OpenMP support
gives a single-threaded build.)*

**Ubuntu / Debian:**

```bash
sudo apt install cmake libeigen3-dev libcereal-dev libboost-all-dev libopenblas-dev libomp-dev
```

**Windows (vcpkg):**
Install `vcpkg`, set `VCPKG_ROOT`, then let the preset handle everything (dependencies are
resolved from [vcpkg.json](vcpkg.json)):

```bash
cmake --workflow --preset vcpkg-release
```

---

## Building the Library and Tests

All commands run from this `cpp/` directory. The presets are defined in
[CMakePresets.json](CMakePresets.json).

```bash
# Debug build (symbols, no optimisation)
cmake --preset debug
cmake --build --preset debug-build

# Release build (optimised, -march=native)
cmake --preset release
cmake --build --preset release-build
```

Useful configure-time options:

| Option                 | Default             | Effect                                                          |
|------------------------|---------------------|-----------------------------------------------------------------|
| `TREX_USE_BLAS_LAPACK` | ON (OFF on Windows) | Route Eigen through a BLAS/LAPACK backend (Accelerate/OpenBLAS) |
| `TREX_PORTABLE_BUILD`  | OFF                 | Disable `-march=native` for portable/wheel/CI builds            |
| `TREX_BUILD_TESTS`     | ON                  | Build the GoogleTest executables                                |
| `TREX_WITH_OPENMP`     | ON                  | Use OpenMP when found. OFF forces the single-threaded build     |

**Running the tests:**

After a successful build, all test executables are placed in `build/{debug,release}/bin/`.
Run the full suite via CTest:

```bash
cd build/debug && ctest --output-on-failure
```

Or run individual executables directly (e.g. to test the core selector):

```bash
./build/debug/bin/test_trex_core
./build/debug/bin/test_tsolvers_execution
```

**Sanitizer and static-analysis presets** for development:

```bash
cmake --workflow --preset asan-ubsan   # AddressSanitizer + UBSan build & tests
cmake --workflow --preset tidy        # clang-tidy over the codebase
```

---

## Installing and Consuming the Library

The library installs CMake package config files and can be consumed by other C++ projects
via `find_package(TRexSelector)`.

### Install the library

**Option A — Install to the project-local `install/` directory:**

```bash
# 1. Configure and build (portable release, no -march=native)
cmake --workflow --preset local-install

# 2. Run the install step
cmake --install build/release-portable
# Installs to cpp/install/ (prefix set in the local-install preset)
```

**Option B — Install system-wide (`/usr/local`):**

```bash
# 1. Configure and build
cmake --workflow --preset release-libs

# 2. Install (sudo required on macOS)
sudo cmake --install build/release-portable --prefix /usr/local
```

**Option C — Install to a custom prefix:**

```bash
cmake --workflow --preset release-libs
cmake --install build/release-portable --prefix /your/custom/prefix
```

### Consume in a downstream project

```cmake
# CMakeLists.txt of the consuming project
find_package(TRexSelector REQUIRED)

target_link_libraries(my_app PRIVATE
    TRexSelector::trex_selector_methods   # pulls in everything transitively
)
```

If the library is not on the standard search path, pass its prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/TRexSelector/cpp/install
```

**Exported targets:**

| Target                                | Contains                                                       |
|---------------------------------------|----------------------------------------------------------------|
| `TRexSelector::trex_selector_methods` | All selectors — links tsolvers, ml\_methods, utils             |
| `TRexSelector::trex_tsolvers`         | T-algorithm solvers only                                       |
| `TRexSelector::trex_ml_methods`       | Standardization, clustering, model selection                   |
| `TRexSelector::trex_utils`            | Foundational utilities (memmap, serialization, OpenMP, Eigen)  |

All targets bring their transitive dependencies (Eigen3, Cereal, OpenMP, Boost, BLAS/LAPACK)
with them — downstream projects do not need to call `find_package` for those individually.
OpenMP is among them only when the library was built with it. `find_package(TRexSelector)`
sets `TRexSelector_WITH_OPENMP` to tell which.

---

## What the Library Provides

The sources under `src/` follow the four exported targets. The R and Python packages bind the
augmented-dummy family of selectors, the T-solvers, the ML methods and the utilities, with the same
names where the language allows it.

### Selectors (`src/trex_selector_methods/`)

The augmented-dummy family, `ad/`, is the T-Rex method family of the papers. Each selector takes its
control struct, which carries the T-solver choice, the dummy settings and the thread budgets.

| Selector | Task | Control |
| --- | --- | --- |
| `TRexSelector` | FDR-controlled variable selection, the T-Rex selector | `TRexControlParameter` |
| `TRexDASelector` | dependency-aware (DA) dummies for correlated variables, T-Rex+DA | `TRexDAControlParameter` |
| `TRexGVSSelector` | group variable selection (GVS), T-Rex GVS | `TRexGVSControlParameter` |
| `ScreenTRexSelector` | FDR-controlled screening of many variables, Screen-T-Rex | `ScreenTRexControlParameter` |
| `BiobankScreenTRex` | phenotype screening of genomic biobank designs | `BiobankScreenTRexControl` |
| `TRexSPCA` | sparse principal component analysis (SPCA) under FDR control | `TRexSPCAControlParameter` |
| `TRexTikhonovSelector` | selection with a general Tikhonov penalty matrix | `TRexTikhonovControlParameter` |

Two further families live next to it and are not bound in R or Python yet. The virtual-dummy family,
`vd/`, holds `TRexVD` with its own solvers (`VD_LARS`, `VD_OMP`, `VD_AFS`, `VD_AFS_Logistic` under
`src/tsolvers/vd/`), ported from the virtual-dummies project of T. Koka, J. Machkour, D. P. Palomar
and M. Muma (arXiv:2604.07464, see the references). The sparse-dummy family, `sd/`, holds `TRexSD`
with the solvers `SD_TLARS_Solver`, `SD_TOMP_Solver` and `SD_TAFS_Solver` under `src/tsolvers/sd/`,
and is future work, as its notes in `trex_sd.md` say.

### T-solvers (`src/tsolvers/ad/`)

Each random experiment runs a terminating solver, a T-solver, which stops once a given number of
dummies has entered its path. Twenty T-solvers derive from `TSolver_Base` in four families,
`SolverOptions` holds their options, and `TRexControlParameter::solver_type` picks the one a selector
uses.

- Least angle regression (LARS) based: `TLARS_Solver`, `TLASSO_Solver`, `TSTEPWISE_Solver`,
  `TSTAGEWISE_Solver`, `TENET_Solver`, `TENETAug_Solver`, `TIENET_Solver`, `TIENETAug_Solver` and
  `TIKNET_Solver`.
- Matching pursuit based: `TOMP_Solver`, `TGP_Solver`, `TACGP_Solver`, `TMP_Solver`, `TOOLS_Solver`
  and `TNCGMP_Solver`.
- Adaptive forward stepwise: `TAFS_Solver`.
- Coordinate descent (CD) based: `TCDLASSO_Solver`, from which `TCDENET_Solver`, `TCDIENET_Solver`
  and `TCDTIKNET_Solver` derive.

### ML methods (`src/ml_methods/`)

The machine learning (ML) building blocks the selectors share: the scalers `ZScoreScaler` and
`LpNormScaler` (`scaler_methods/`), the decompositions `SVDSolver` (`svd/`) and `PCA` (`pca/`), the
`RidgeSolver` (`ridge_regression/`), the cross-validated model selection of ridge, elastic-net,
informed elastic-net and Tikhonov penalties (`model_selection/`), and `AgglomerativeClustering`
(`clustering/`).

### Utilities (`src/utils/`)

- `memmap/`: `MemoryMappedMatrix`, the disk-backed design the selectors and T-solvers read without a
  copy.
- `eval_metrics/`: `compute_fdp()`, `compute_tpp()`, `compute_precision()` and `compute_recall()` on
  index sets, their dense counterparts on coefficient vectors, and the true-positive, false-positive
  and false-negative counts.
- `openmp/`: `set_num_threads()`, `get_num_threads()`, `get_max_threads()` and `threads_for()`, the
  thread budget of every kernel, with stubs for builds without OpenMP (see Threads).
- `datageneration/`: the dummy, design and noise generators. `serialization/`: the cereal adapters for
  Eigen types. `logging/` and `fp_classify/`: the logger and the floating-point classification.

---

## Dummy Strategies

The T-Rex selector calibrates the number of dummies in its L-loop. `lloop_strategy` of
`TRexControlParameter` says how the dummies of that loop are produced, in every selector that
carries the control.

| Strategy | Dummies | Storage |
| --- | --- | --- |
| `STANDARD` (default) | fresh matrices at every L step | K matrices |
| `HCONCAT` | the same K matrices grow by p columns per L step | K matrices |
| `SEEDED` | the draws of `STANDARD`, re-derived from the seed at every step | none |
| `PERMUTATION` | permuted columns of the design | K matrices |
| `PERMUTATION_SEEDED` | the permutation re-derived from the seed | none |
| `SKIPL` | no L-loop, the cap `max_dummy_multiplier` is used directly | K matrices |

`STANDARD` draws fresh dummies at every step, the law of the T-Rex paper and the default on every
surface. The seeded strategies store nothing and suit designs where K times L times p columns do
not fit in memory. `SEEDED` selects exactly as `STANDARD` for the same seed, and
`PERMUTATION_SEEDED` exactly as `PERMUTATION`. `HCONCAT` keeps the grown law, which calibrates a
larger L and gains 0.6 percentage points of true positive rate (TPR) on average in a paired study
of 500 trials, at the same false discovery rate (FDR). `SKIPL` fixes L at the cap and skips the
calibration, and under a large fixed L without the stagnation stop its FDP estimate is
anti-conservative (the selector warns from `max_dummy_multiplier = 10` on), so the calibrated
strategies are the safe choice. The dummies are standard Normal by default, and
`dummy_distribution` offers the other laws. The enum `LLoopStrategy` in trex.hpp documents the
keys the seeded strategies derive their draws from.

---

## Threads

The library runs serially by default: `TRexControlParameter` has
`parallel_rnd_experiments = false`, `max_outer_threads = 1` and `max_inner_threads = 1`, and
`SolverOptions` has `n_threads = 1`. The random experiments of a selector run one after the
other, every T-solver on one thread. Parallel work is opt-in, and one layer owns the cores at a
time.

* `parallel_rnd_experiments = true` with `max_outer_threads = m` runs the K random experiments
  on `m` threads, at most K. Inside the parallel experiments every solver runs serially.
* `max_inner_threads = m` gives the solvers inside the experiments a budget of `m` threads for
  their passes over the design. The budget is live when the experiments run in sequence, which
  the lean L-loop strategies (SEEDED, PERMUTATION, PERMUTATION_SEEDED) always do. A pass is
  split only when it touches at least a million entries of the design
  (`TSolver_Base::kParallelWorkThreshold`), because the cost of forming a team of threads
  exceeds the gain below that. Two to four threads carry the gain, since the passes are
  memory-bandwidth bound. The seeded generators (the dummies, the synthetic designs) count
  their work in the same unit with the measured cost of a random draw (about 80 entries) and
  of seeding an engine (up to 32,000), so they split from a few thousand entries.
* `SolverOptions::n_threads = m` is the same budget for a T-solver used on its own, and
  `TSolver_Base::setNumThreads()` changes it on a live solver.
* Every other kernel (the scalers, the clustering, the ML regressions, the data normalizer, the
  dummy and data generators) runs on the library's default budget, which starts at 1, so the
  library is single-threaded by default on every machine. `trex::utils::openmp::set_num_threads(m)`
  sets that default for the process and returns the previous value, `get_num_threads()` reads it,
  and the `n_threads` option of the scalers, the clustering and the ML classes overrides it per
  object. Inside a selector these kernels take `max_inner_threads`. `get_max_threads()` reports
  the threads OpenMP can run on the machine, an upper bound for a budget. The OpenMP runtime's
  own default team is never used.

Every thread option is a whole number of at least 1, and other values throw `std::invalid_argument`.
A seeded run selects the same variables whatever the thread settings. Without OpenMP every
setting is a no-op and `get_max_threads()` returns 1.

---

## Relation to the Language Packages

* The Python package ([Python/trexselector](../Python/trexselector/)) compiles this
  source tree directly via scikit-build-core — no separate library install is needed.
* The R package ([R/TRexSelector](../R/TRexSelector/)) vendors a copy of `src/` at
  release time via `tools/bundle_for_cran.sh`, so the CRAN tarball is self-contained.

## License

GPL ≥ 3 — see [LICENSE](LICENSE) or the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.html).

```text
This program is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
