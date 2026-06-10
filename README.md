# TRexSelector

**C++20 library for FDR-controlled variable selection in high-dimensional regression and classification.**

---

## About the T-Rex Selector

The T-Rex Selector is a statistically principled method for selecting relevant predictors from
high-dimensional datasets while rigorously controlling the False Discovery Rate (FDR) — the
expected proportion of falsely selected variables — at a user-specified level.
The algorithm works by generating random dummy variables that act as a calibration baseline:
during each of *K* independent random experiments, a terminating solver traces solution paths
through both the real predictors and the dummies.
By comparing how often each real predictor is selected relative to the dummies across all
experiments, a voting threshold is derived that provably bounds the FDR.
The result is a fast selector that scales to high-dimensional problems without
sacrificing statistical guarantees.

**Papers:**

- [Machkour, Muma, Palomar — *The Terminating-Random Experiments Selector: Fast High-Dimensional Variable Selection with False Discovery Rate Control*, Elsevier Signal Processing (2025)](https://www.sciencedirect.com/science/article/pii/S016516842500009X)
- [Machkour, Muma, Palomar - *High-dimensional false discovery rate control for dependent variables*, Elsevier Signal Processing (2025)](https://www.sciencedirect.com/science/article/pii/S0165168425001045)

**Project website:** [trexselector.com](https://trexselector.com)

---

## Implemented Variants

| Class | Description |
|---|---|
| `TRexSelector` | Core FDR-controlled selector for linear regression |
| `TRexDASelector` | Dependency-aware variant for correlated predictors |
| `TRexGVSSelector` | Grouped variable selection |
| `TRexScreeningSelector` | Fast FDR-controlled variable pre-screening for large *p* |
| `TRexBiobankScreeningSelector` | High-throughput genomic / biobank screening |

R and Python bindings expose all variants — see the [R Package](#r-package) and
[Python Package](#python-package) sections below.

---

## Dependencies

| Library | Role | Version | Required |
|---|---|---|---|
| [Eigen3](https://eigen.tuxfamily.org) | Linear algebra | any recent | Yes |
| [Boost](https://www.boost.org) | Headers + iostreams (memory-mapped files) | ≥ 1.80 | Yes |
| [Cereal](https://uscilab.github.io/cereal) | Serialization of solver state | any recent | Yes |
| [OpenMP](https://www.openmp.org) | Parallelisation of random experiments | ≥ 3.1 | Yes |
| BLAS / LAPACK | Accelerated linear algebra backend | — | Optional |

On **macOS**, BLAS/LAPACK defaults to Apple Accelerate. On **Linux**, OpenBLAS is used.
The BLAS/LAPACK backend can be disabled at configure time with `-DTREX_USE_BLAS_LAPACK=OFF`.

---

## Quick Install

**macOS (Homebrew):**

```bash
brew install llvm eigen cereal boost openblas libomp
```

> CMake requires Homebrew LLVM (not Apple Clang) for OpenMP support.
> The build system detects it automatically at `/opt/homebrew/opt/llvm`.

**Ubuntu / Debian:**

```bash
sudo apt install libeigen3-dev libcereal-dev libboost-all-dev libopenblas-dev libomp-dev
```

**Windows (vcpkg):**

Install [vcpkg](https://github.com/microsoft/vcpkg), set `VCPKG_ROOT`, then let the
preset handle everything:

```bash
cmake --workflow --preset vcpkg-release
```

vcpkg reads `vcpkg.json` and automatically installs Eigen3, Boost, Cereal, and libomp.

---

## Build

Configure and build using CMake presets:

```bash
# Debug build (symbols, no optimisation)
cmake --preset debug
cmake --build --preset debug-build

# Release build (optimised)
cmake --preset release
cmake --build --preset release-build
```

Or use the VS Code tasks: **CMake Full Build (Debug)** / **CMake Full Build (Release)**
via *Terminal → Run Task…*

Build artefacts land in `build/debug/` and `build/release/`. Executables are placed in
the `bin/` subdirectory of each build directory.

> **macOS pitfall:** If IntelliSense or builds break after a branch switch or library upgrade,
> see [notes/IDE_Troubleshooting.md](notes/IDE_Troubleshooting.md) for the full
> cache-clearing and reconfiguration procedure.

---

## Testing

The test suite uses GoogleTest. After building:

```bash
# Debug
cd build/debug && ctest --output-on-failure

# Release
cd build/release && ctest --output-on-failure
```

---

## R Package

The R bindings live in [`R/TRexSelector/`](R/TRexSelector/) and are implemented with
RcppEigen. The C++ backend is compiled automatically on installation — no separate
CMake step is needed.

**Install from source:**

```r
install.packages("path/to/R/TRexSelector", repos = NULL, type = "source")
```

Or directly from the repository root:

```bash
R CMD INSTALL R/TRexSelector
```

CRAN submission is in progress. The package passes `R CMD check --as-cran` with no
errors, warnings, or notes on macOS arm64.

---

## Python Package

The Python bindings live in [`Python/TRexSelector/`](Python/TRexSelector/) and are
implemented with pybind11, built via scikit-build-core.

**Install from source** (requires CMake ≥ 3.24, a C++20 compiler, and the C++ dependencies above):

```bash
pip install --no-build-isolation -e Python/TRexSelector
```

PyPI distribution is planned. See
[notes/Plan for Packages.txt](notes/Plan%20for%20Packages.txt) for the release roadmap.

---

## Platform Compatibility

| Platform | Architecture | C++ Core | R Package | Python Package |
|---|---|---|---|---|
| macOS | arm64 | Tested ✓ | Tested ✓ | Tested ✓ |
| Linux | x86_64 | Open | Open | Open |
| Windows | x86_64 | Open | Open | Open |

Linux and Windows support is expected to work but has not yet been formally verified.
Contributions and test reports are welcome.

---

## For C++ Developers

The library exports CMake targets and can be consumed by other C++ projects.

**Install:**

```bash
cmake --install build/release --prefix /usr/local
```

This installs headers to `/usr/local/include/` and the CMake config files to
`/usr/local/lib/cmake/TRexSelector/`.

**Consume in your project:**

```cmake
find_package(TRexSelector REQUIRED)
target_link_libraries(my_app PRIVATE trex::trex_selector_methods)
```

Available targets: `trex::utils`, `trex::ml_methods`, `trex::tsolvers`,
`trex::trex_selector_methods`.

---

## References

- [Machkour, Muma, Palomar — *The Terminating-Random Experiments Selector: Fast High-Dimensional Variable Selection with False Discovery Rate Control*, Elsevier Signal Processing (2025)](https://www.sciencedirect.com/science/article/pii/S016516842500009X)
- [Machkour, Muma, Palomar - *High-dimensional false discovery rate control for dependent variables*, Elsevier Signal Processing (2025)](https://www.sciencedirect.com/science/article/pii/S0165168425001045)

---

## License

GPL ≥ 3 — see [LICENSE](LICENSE).
