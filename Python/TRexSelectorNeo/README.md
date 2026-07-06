# TRexSelectorNeo

[![PyPI version](https://badge.fury.io/py/TRexSelectorNeo.svg)](https://badge.fury.io/py/TRexSelectorNeo)

**Title**: The T-Rex selector for fast high-dimensional variable selection with FDR control

**Description**: It performs fast variable selection in large-scale high-dimensional settings while controlling the false discovery rate (FDR) at a user-defined target level. The Python package provides bindings to the TRexSelector C++ library via pybind11. It is distributed as `TRexSelectorNeo` and imported as `trex_selector_neo` — distinct from the earlier pure-Python `trexselector` package, which it supersedes.

**Papers**: The package is based on the papers

J. Machkour, M. Muma, and D. P. Palomar, "The terminating-random experiments selector: Fast high-dimensional variable selection with false discovery rate control," *Signal Processing*, vol. 231, article 109894, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109894>)

J. Machkour, M. Muma, and D. P. Palomar, "High-dimensional false discovery rate control for dependent variables," *Signal Processing*, vol. 234, article 109990, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109990>)

---

## Installation

**From PyPI** — binary wheels are provided for Linux (x86_64), macOS (arm64), and Windows (x86_64), so no compiler is needed:

```bash
pip install TRexSelectorNeo
```

Requirements: Python ≥ 3.10 and NumPy ≥ 1.26 (installed automatically).

**From source (GitHub)** — builds the C++ backend on your machine; requires a C++20 compiler, CMake ≥ 3.24, and the C++ dependencies Eigen3, Boost ≥ 1.80, Cereal, and OpenMP (see the [C++ core README](../../cpp/README.md) for how to install them per platform):

```bash
pip install git+https://github.com/G4Lactus/TRexSelector.git#subdirectory=Python/TRexSelectorNeo
```

**Development install** — from a clone of the repository, as an editable install without build isolation (pybind11, scikit-build-core, and CMake must already be available in the active environment):

```bash
pip install --no-build-isolation -e Python/TRexSelectorNeo

# Run the test suite against the installed package
pytest Python/TRexSelectorNeo/tests -q
```

---

## Quick Start

```python
import numpy as np
import trex_selector_neo

rng = np.random.default_rng(123)
n, p, num_act = 75, 150, 3        # observations, variables, true actives
beta = np.zeros(p)
beta[:num_act] = 1.0              # coefficient vector with sparse support
true_actives = np.nonzero(beta)[0].tolist()  # 0-based indices

X = rng.standard_normal((n, p))
y = X @ beta + rng.standard_normal(n)

sel = trex_selector_neo.TRexSelector(X, y, tFDR=0.05, verbose=False)
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

## Links

T-Rex paper: <https://doi.org/10.48550/arXiv.2110.06048>

TRexSelector package: [GitHub](https://github.com/G4Lactus/TRexSelector)

Bug reports: [GitHub Issues](https://github.com/G4Lactus/TRexSelector/issues)
