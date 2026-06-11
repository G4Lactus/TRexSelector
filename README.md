# THE T-REX SELECTOR

**FDR-controlled variable selection in high-dimensional regression and classification. A
C++20 library with R and Python bindings.**

---

## A Short Story about the T-Rex Selector

False discoveries plague science and engineering.
In genomics, a single false discovery can wrongly tie a gene to a disease, contributing to the
reproducibility crisis and derailing years of research and clinical effort [(Huffman et al., 2018)](https://www.nature.com/articles/s41467-018-07348-x).
In finance, hundreds of studies claim to have found factors that explain stock returns, yet most of
these findings are likely false [(Harvey, Liu, and Zhu, 2016)](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824?redirectedFrom=fulltext).

The T-Rex Selector performs high-dimensional variable selection while controlling the rate of false
discoveries at a level you choose, with rigorous finite-sample guarantees, and fast and
memory-efficient enough to run on millions of variables [(Machkour, Muma, and Palomar, 2025)](https://www.sciencedirect.com/science/article/pii/S016516842500009X).

The algorithm works by generating random dummy variables that act as a calibration baseline:
during each of *K* independent random experiments, a terminating solver traces solution paths
through both the real predictors and the dummies.
By comparing how often each real predictor is selected relative to the dummies across all
experiments, a voting threshold is derived that provably bounds the FDR.

**Project website:** [trexselector.com](https://trexselector.com)

---

## Implemented Variants

| Class | Description |
|-------|-------------|
| `TRexSelector` | Core FDR-controlled selector for linear regression |
| `TRexDASelector` | Dependency-aware variant for correlated predictors |
| `TRexGVSSelector` | Grouped variable selection |
| `TRexScreeningSelector` | Fast FDR-controlled variable pre-screening for large *p* |
| `TRexBiobankScreeningSelector` | High-throughput genomic / biobank screening |

---

## Dependencies & Platform Compatibility

| Library | Role | Version | Required |
|---------|------|---------|----------|
| [Eigen3](https://eigen.tuxfamily.org) | Linear algebra | any recent | Yes |
| [Boost](https://www.boost.org) | Headers + iostreams (memory-mapped files) | ≥ 1.80 | Yes |
| [Cereal](https://uscilab.github.io/cereal) | Serialization of solver state | any recent | Yes |
| [OpenMP](https://www.openmp.org) | Parallelisation of random experiments | ≥ 3.1 | Yes |
| BLAS / LAPACK | Accelerated linear algebra backend | — | Optional |

**Platform Support:**

* **macOS (arm64):** Fully tested (C++ Core, R, and Python packages). BLAS/LAPACK defaults to Apple Accelerate.
* **Linux (x86_64):** Supported (OpenBLAS is used).
* **Windows (x86_64):** Supported.

*The BLAS/LAPACK backend can be disabled at configure time with `-DTREX_USE_BLAS_LAPACK=OFF`.*

---

## C++ Core Installation

**macOS (Homebrew):**

```bash
brew install llvm eigen cereal boost openblas libomp
```

*(CMake requires Homebrew LLVM, not Apple Clang, for OpenMP support).*

**Ubuntu / Debian:**

```bash
sudo apt install libeigen3-dev libcereal-dev libboost-all-dev libopenblas-dev libomp-dev
```

**Windows (vcpkg):**
Install `vcpkg`, set `VCPKG_ROOT`, then let the preset handle everything:

```bash
cmake --workflow --preset vcpkg-release
```

**Build Commands:**

```bash
# Debug build (symbols, no optimisation)
cmake --preset debug
cmake --build --preset debug-build

# Release build (optimised)
cmake --preset release
cmake --build --preset release-build
```

---

## Language Ecosystems

### R Package

The R bindings live in `R/TRexSelector/` and are implemented with RcppEigen. The C++ backend is compiled automatically on installation.

**Install from source:**

```bash
R CMD INSTALL R/TRexSelector
```

*(CRAN submission is in progress. The package passes `R CMD check --as-cran` cleanly).*

### Python Package

The Python bindings live in `Python/TRexSelector/` and are implemented with pybind11, built via scikit-build-core.

**Install from source:**
*(Requires CMake ≥ 3.24, a C++20 compiler, and the C++ dependencies above)*

```bash
pip install --no-build-isolation -e Python/TRexSelector
```

---

## For C++ Developers

The library installs CMake package config files and can be consumed by other C++ projects via `find_package(TRexSelector)`.

**Step 1 — Configure and build:**

```bash
# Debug build (symbols, no optimisation)
cmake --preset debug
cmake --build --preset debug-build

# Release build (optimised)
cmake --preset release
cmake --build --preset release-build
```

**Step 2 — Install to a prefix:**

```bash
cmake --install build/release --prefix /path/to/install
```

**Step 3 — Consume in your project (`CMakeLists.txt`):**

```cmake
find_package(TRexSelector REQUIRED)

target_link_libraries(my_app PRIVATE
    TRexSelector::trex_selector_methods   # pulls in everything transitively
)
```

Available targets:

| Target | Contains |
|--------|----------|
| `TRexSelector::trex_selector_methods` | All selectors — links tsolvers, ml_methods, utils |
| `TRexSelector::trex_tsolvers` | T-algorithm solvers only |
| `TRexSelector::trex_ml_methods` | Standardization, clustering, model selection |
| `TRexSelector::trex_utils` | Foundational utilities (memmap, serialization, OpenMP, Eigen) |

Point `CMAKE_PREFIX_PATH` at your install prefix so `find_package` can locate the config:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install
```

---

## References

* Machkour, Muma, Palomar - *The Terminating-Random Experiments Selector: Fast High-Dimensional Variable Selection with False Discovery Rate Control*, Elsevier Signal Processing (2025). [DOI link](https://www.sciencedirect.com/science/article/pii/S016516842500009X)
* Machkour, Muma, Palomar - *High-dimensional false discovery rate control for dependent variables*, Elsevier Signal Processing (2025). [DOI link](https://www.sciencedirect.com/science/article/pii/S0165168425001045)
* Huffman, J. E., et al. - *Examining the current standards for genetic discovery and replication in the era of mega-biobanks*, Nature Communications, 9(1), 5054 (2018). [DOI link](https://www.nature.com/articles/s41467-018-07348-x)
* Harvey, C. R., Liu, Y., and Zhu, H. - *… and the Cross-Section of Expected Returns*, The Review of Financial Studies, 29(1), 5–68 (2016). [DOI link](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824?redirectedFrom=fulltext)

---

## License

GPL ≥ 3 — see [LICENSE](LICENSE).
