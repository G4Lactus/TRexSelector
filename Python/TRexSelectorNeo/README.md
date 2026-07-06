# TRexSelectorNeo

[![PyPI version](https://badge.fury.io/py/TRexSelectorNeo.svg)](https://badge.fury.io/py/TRexSelectorNeo)

**Title**: The T-Rex selector for fast high-dimensional variable selection with FDR control

**Description**: It performs fast variable selection in large-scale high-dimensional settings while controlling the false discovery rate (FDR) at a user-defined target level. The Python package provides bindings to the TRexSelector C++ library via pybind11. It is distributed as `TRexSelectorNeo` and imported as `trex_selector_neo` — distinct from the earlier pure-Python `trexselector` package, which it supersedes.

**Papers**: The package is based on the papers

J. Machkour, M. Muma, and D. P. Palomar, "The terminating-random experiments selector: Fast high-dimensional variable selection with false discovery rate control," *Signal Processing*, vol. 231, article 109894, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109894>)

J. Machkour, M. Muma, and D. P. Palomar, "High-dimensional false discovery rate control for dependent variables," *Signal Processing*, vol. 234, article 109990, 2025. (<https://doi.org/10.1016/j.sigpro.2025.109990>)

---

## Requirements

- Python >= 3.10, NumPy >= 1.26
- C++20 compiler, CMake >= 3.24, Eigen3, Boost, Cereal, OpenMP — only when building from source

---

## Installation

```bash
pip install TRexSelectorNeo
# or from GitHub:
pip install git+https://github.com/G4Lactus/TRexSelector.git#subdirectory=Python/TRexSelector
```

---

## Quick Start

```python
import numpy as np
import trex_selector_neo

rng = np.random.default_rng(123)
n, p, num_act = 75, 150, 3
beta = np.zeros(p); beta[:num_act] = 1.0
true_actives = list(np.nonzero(beta)[0])  # 0-based

X = rng.standard_normal((n, p))
y = X @ beta + rng.standard_normal(n)

sel = trex_selector_neo.TRexSelector(X, y, tFDR=0.05, verbose=False)
sel.select()

print(f"True active variables: {true_actives}")
print(f"Selected variables:    {sel.selected_indices}")
```

---

## Links

T-Rex paper: <https://doi.org/10.48550/arXiv.2110.06048>

TRexSelector package: [GitHub](https://github.com/G4Lactus/TRexSelector)

Bug reports: [GitHub Issues](https://github.com/G4Lactus/TRexSelector/issues)
