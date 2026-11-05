# trexselector

[![PyPI version](https://badge.fury.io/py/trexselector.svg)](https://badge.fury.io/py/trexselector)

**Title**: The T-Rex selector for fast high-dimensional variable selection with FDR control

**Description**: It performs fast variable selection in large-scale high-dimensional settings while controlling the
false discovery rate (FDR) at a user-defined target level. The Python package provides bindings to the TRexSelector C++
library via pybind11. It is distributed on PyPI and imported as `trexselector`. Version 1.0.0 replaces the pure-Python
implementation of the 0.6.x releases under the same package name.

**Papers**: The package is based on the papers

J. Machkour, M. Muma, and D. P. Palomar, "The terminating-random experiments selector: Fast high-dimensional variable selection with false discovery rate control," *Signal Processing*, vol. 231, article 109894, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109894>)

J. Machkour, M. Muma, and D. P. Palomar, "High-dimensional false discovery rate control for dependent variables," *Signal Processing*, vol. 234, article 109990, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109990>)

---

## Installation

**From PyPI** — binary wheels are provided for Linux (x86_64), macOS (arm64), and Windows (x86_64), so no compiler is
needed:

```bash
pip install trexselector
```

Requirements: Python ≥ 3.10 and NumPy ≥ 1.26 (installed automatically).

**From source (GitHub)** — builds the C++ backend on your machine; requires a C++20 compiler, CMake ≥ 3.24, and the C++
dependencies Eigen3, Boost ≥ 1.80 and Cereal, with OpenMP optional (see the [C++ core README](../../cpp/README.md) for
how to install them per platform):

```bash
pip install git+https://github.com/trexselector/TRexSelector.git#subdirectory=Python/trexselector
```

Without OpenMP the package builds single-threaded and `trexselector.utils.get_max_threads()` returns 1. The CMake
switch `TREX_WITH_OPENMP=OFF` forces that build where OpenMP exists:

```bash
pip install -C cmake.define.TREX_WITH_OPENMP=OFF Python/trexselector
```

**Development install** — from a clone of the repository, as an editable install without build isolation (pybind11,
scikit-build-core, and CMake must already be available in the active environment):

```bash
pip install --no-build-isolation -e Python/trexselector

# Run the test suite against the installed package
pytest Python/trexselector/tests -q
```

---

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

For a preset target FDR of 5%, the T-Rex selector recovers all true active variables with no
false positives in this example. Choose the target FDR (`tFDR`) according to the requirements
of your application.

All selector variants (`TRexSelector`, `TRexDASelector`, `TRexGVSSelector`,
`TRexScreeningSelector`, `TRexBiobankScreeningSelector`, `TRexSPCASelector`) follow the same
construct → `select()` → inspect pattern.

---

## Dummy Strategies

The T-Rex selector calibrates the number of dummies in its L-loop. `TRexControlParameter.lloop_strategy` says how the
dummies of that loop are produced, in every selector that carries the control, with the `LLoopStrategy` enum.

| Strategy | Dummies | Storage |
| --- | --- | --- |
| `STANDARD` (default) | fresh matrices at every L step | K matrices |
| `HCONCAT` | the same K matrices grow by p columns per L step | K matrices |
| `SEEDED` | the draws of `STANDARD`, re-derived from the seed at every step | none |
| `PERMUTATION` | permuted columns of the design | K matrices |
| `PERMUTATION_SEEDED` | the permutation re-derived from the seed | none |
| `SKIPL` | no L-loop, the cap `max_dummy_multiplier` is used directly | K matrices |

`STANDARD` draws fresh dummies at every step, the law of the T-Rex paper and the default on every surface. The seeded
strategies store nothing and suit designs where K times L times p columns do not fit in memory. `SEEDED` selects exactly
as `STANDARD` for the same seed, and `PERMUTATION_SEEDED` exactly as `PERMUTATION`. `HCONCAT` keeps the grown law, which
calibrates a larger L and gains 0.6 percentage points of true positive rate (TPR) on average in a paired study of 500
trials, at the same false discovery rate (FDR). `SKIPL` fixes L at the cap and skips the calibration, and under a large
fixed L without the stagnation stop its FDP estimate is anti-conservative (the selector warns from `max_dummy_multiplier
= 10` on), so the calibrated strategies are the safe choice. The dummies are standard Normal by default, and
`dummy_distribution` offers the other laws. The R vignette page 2 carries the study.

---

## Threads

The package runs serially by default: `TRexControlParameter` has `parallel_rnd_experiments = False`, `max_outer_threads
= 1` and `max_inner_threads = 1`, and `SolverOptions` has `n_threads = 1`. The random experiments of a selector run one
after the other, every T-solver on one thread. Parallel work is opt-in, and one layer owns the cores at a time.

- `parallel_rnd_experiments = True` with `max_outer_threads = m` runs the K random experiments on `m` threads, at most
  K. Inside the parallel experiments every solver runs serially.
- `max_inner_threads = m` gives the solvers inside the experiments a budget of `m` threads for their passes over the
  design. The budget is live when the experiments run in sequence, which the lean L-loop strategies (`SEEDED`,
  `PERMUTATION`, `PERMUTATION_SEEDED`) always do. A pass is split only when it touches at least a million entries of the
  design, because the cost of forming a team of threads exceeds the gain below that. Two to four threads carry the gain,
  since the passes are memory-bandwidth bound. The seeded generators (the dummies, the synthetic designs) count their
  work in the same unit with the measured cost of a random draw (about 80 entries) and of seeding an engine (up to
  32,000), so they split from a few thousand entries.
- `SolverOptions(n_threads=m)` is the same budget for a T-solver used on its own, and `setNumThreads()` changes it on a
  live solver.
- Every other kernel (the scalers, the clustering, the ML regressions, `normalize_design()`, the dummy and data
  generators) runs on the library's default budget, which starts at 1, so the package is single-threaded by default on
  every machine. `trexselector.utils.set_num_threads(m)` sets that default for the process and returns the previous
  value, `get_num_threads()` reads it, and the `n_threads` argument of the scalers, the clustering and the ML classes
  overrides it per object. Inside a selector these kernels take `max_inner_threads`. `get_max_threads()` reports the
  threads OpenMP can run on the machine, an upper bound for a budget.

Every thread option is a whole number of at least 1; other values raise `ValueError`. A seeded run selects the same
variables whatever the thread settings. Without OpenMP every setting is a no-op and `get_max_threads()` returns 1 (see
Installation).

---

## Links

T-Rex paper: <https://doi.org/10.1016/j.sigpro.2025.109894>

TRexSelector package: [GitHub](https://github.com/trexselector/TRexSelector)

Bug reports: [GitHub Issues](https://github.com/trexselector/TRexSelector/issues)

Citation metadata: [CITATION.cff](../../CITATION.cff) at the repository root. The docstring of every selector,
T-solver and machine learning class ends with the papers it builds on.

## Acknowledgements

The library builds on the Eigen linear algebra library: G. Guennebaud, B. Jacob et al., Eigen v3,
<https://libeigen.gitlab.io/>, 2010. It binds the C++ core through pybind11 and relies on cereal for
serialization, on Boost.Interprocess for memory mapping and on OpenMP for the threads, with an optional
BLAS/LAPACK backend.

## License

GPL ≥ 3 — see [LICENSE](../../LICENSE).

This program is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
