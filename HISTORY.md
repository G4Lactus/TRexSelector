# TRexSelector changelog

## TRexSelector ?.?.?

????-??-??

### 2026-07-05

#### T-solver review fixes (tsolvers module)

- Full code review of `cpp/src/tsolvers/` with correctness fixes: stale
  correlations after T-Stagewise NNLS drops, missing dummy de-normalization in
  T-ENET's `getBeta`/`getBetaPath`, a data race and a scale-aware
  near-constant-column drop threshold in the preprocessing, guarded
  `getBeta(step)`/`getCp()` accessors, and a consistent
  actives/inactives/dropped partition across all solvers. `restore()` no
  longer corrupts the caller's `y` (the response is copied internally and was
  never centered in place).
- API: `executeStep()` with `T_stop = 0` now computes the full solution path as
  documented; `setTieSeed()` and `getWarnings()`/`clearWarnings()` are public
  and exposed in Python; warnings print unconditionally and are recorded
  per solver; TENETAug mirrors its inner solver's diagnostics, validates
  `lambda2`, and supports `save()`/`load()`.
- Performance: RAII guard restores Eigen's global thread count after solver
  runs; beta-path storage grows geometrically instead of per-step; dead
  Cholesky backup copies removed.
- T-ACGP now applies the conjugate correction across active-set growth steps
  via zero-padded previous directions (Blumensath & Davies 2008).
- Internals deduplicated (correlation refresh, tie scans, save/load,
  cycling-ratio tracking); tie-breaking RNG state, tolerance, and warnings are
  serialized. NOTE: the checkpoint format changed — `.bin` files saved with
  earlier builds cannot be loaded.

#### T-Rex selector review fixes (trex_selector_methods module)

- Warm starts now actually engage: `WarmStartManager` never stored solvers or
  checkpoint paths, so every T-loop step silently re-solved from scratch. The
  manager now retains solvers in-memory (co-retaining the dummy matrix for
  the PERMUTATION/DIRECT strategies, whose D_k is a per-call temporary) and
  registers serialized checkpoints for the memory-mapped path; `invalidate()`
  keeps the K slots repopulatable instead of shrinking them away. The T-loop
  now continues each solver's path incrementally, as in the R reference;
  results are unchanged (warm/cold equality is asserted by new tests). The
  checkpoint temp directory is only created for serialized mode.
- DIRECT / PERMUTATION_DIRECT dummy generation honours the user seed: it
  previously derived from the PERMUTATION-only base seed (always 0, so
  different seeds produced identical dummies) and, when unseeded, drew fresh
  `random_device` entropy on every call — changing the dummies between
  T-steps. The base seed is now resolved once per `DummyGenerator` and
  reused, making unseeded runs internally consistent and seeded runs
  reproducible across instances.
- Documentation: the intentional `Phi_prime` [0, 1] clamp (deviation from the
  R reference / paper to avoid negative FDP estimates) now documents its
  conservative/anti-conservative asymmetry; the DA-BT deflation cites
  Definition 1 of the T-Rex+DA paper (empty-group penalty Ψ = 1/2) and the
  deliberate R-parity choice of terminal-Φ deltas vs. the paper's per-step
  penalties (Eq. (10)).
- Tests: warm-start engagement and warm/cold equality (in-memory, DIRECT, and
  an end-to-end memory-mapped vs. in-memory run that asserts the T-loop
  actually continues), DIRECT seed handling; fixed a checkpoint filename race
  between the typed serialization tests under parallel ctest.
- Cleanups from the review follow-up: the voting-grid sentinel uses machine
  epsilon instead of the solver tolerance (a large user tol with K >= 100
  could make the grid non-monotone; also matches the R reference grid); the
  AR1 window fallback for an exactly-zero correlation estimate is the
  formula's limit (±1 window) instead of the full window; the
  equi-correlation estimator excludes near-constant columns from the pair
  count; warm-start solvers are invalidated before the dummy matrices they
  view are mutated (STANDARD/HCONCAT), with the buffer-reallocation contract
  documented on the DummyGenerator; dead validation branches removed; the
  Yule-Walker vs. `arima(..., method="ML")` auto-estimation difference for
  the AR1 `cor_coef` is documented on the parameter and covered by a
  known-rho recovery test.

## TRexSelector 2.0.0

### 2026-06-11

#### Architecture

- Complete architectural rework: C++ core library extracted into a standalone
  CMake project (`TRexSelector`) with full `find_package()` support; R and
  Python bindings are now thin language layers on top.
- C++20 standard adopted throughout; Eigen3, Boost, Cereal, and OpenMP are
  managed via vcpkg or system package managers.
- Simulation and demo executables separated into a standalone sibling project
  (`TRexSelector_Simulations`); the core library no longer depends on Matplot++.

#### Selector variants

- `TRexSelector` — base FDR-controlled selector for linear regression.
- `TRexDASelector` — dependency-aware variant for correlated predictors.
- `TRexGVSSelector` — grouped variable selection.
- `TRexScreeningSelector` — fast FDR-controlled variable pre-screening for
  large *p*.
- `TRexBiobankScreeningSelector` — high-throughput genomic / biobank screening.

#### T-algorithm solvers

- LARS-based family: T-LARS, T-LASSO, T-ENET, T-Stagewise, T-Stepwise, TIENET.
- OMP-based family: T-OMP, T-GP, T-ACGP, T-MP, T-NCGMP, T-OLS.
- AFS-based family: T-AFS.
- All solvers support Cereal polymorphic serialization for warm-start workflows.

#### CMake / build system

- Four exported CMake targets:
  `TRexSelector::trex_utils`, `TRexSelector::trex_ml_methods`,
  `TRexSelector::trex_tsolvers`, `TRexSelector::trex_selector_methods`.
- `cmake --install` produces a fully relocatable install tree consumed via
  `find_package(TRexSelector REQUIRED)`.
- `TREX_PORTABLE_BUILD` option disables `-march=native` for portable wheel
  builds (PyPI / CI).
- `TREX_USE_BLAS_LAPACK` option controls BLAS/LAPACK backend (default ON;
  disabled automatically for Python wheel builds).
- CMakePresets.json provides `debug`, `release`, `release-libs`, and
  `vcpkg-release` presets.

#### R package (version 2.0.0)

- New R6 object-oriented API replaces the previous functional `trex()` API.
- C++ backend is compiled via Rcpp/RcppEigen; no longer depends on the `tlars`
  R package.
- R6 classes: `TRexSelector`, `TRexDASelector`, `TRexGVSSelector`,
  `TRexScreeningSelector`, `TRexBiobankScreeningSelector`.
- Helper functions: `compute_fdp()`, `compute_tpp()`.
- Full testthat test suite and a comprehensive vignette added.
- `R CMD check --as-cran` passes with 0 errors, 0 warnings, 1 NOTE (private
  GitHub URL, expected for new submission).
- Compiler flags set to `-O3 -funroll-loops` in Makevars.in / Makevars.win for
  improved runtime performance.

#### Python package (version 0.1.2)

- pybind11 bindings built via scikit-build-core.
- API mirrors the R6 interface; indices are 0-based.
- Classes: `TRexSelector`, `TRexDASelector`, `TRexGVSSelector`,
  `TRexScreeningSelector`, `TRexBiobankScreeningSelector`.
- 355 pytest tests covering all selectors, all solvers, ML utilities, and
  memory-boundary correctness.
- GIL released during long-running C++ computations.
- `pyproject.toml` contains full PyPI metadata (authors, classifiers, URLs).

#### Documentation

- Root `README.md` updated with C++ developer installation guide, full
  `find_package` consume instructions, and published Elsevier paper references.
- R `README.Rmd` and vignette updated with published journal references
  (Signal Processing, 2025).
- `inst/CITATION` lists both published papers.

---

## TRexSelector 0.1.0

### 2026-05-11

First installable release of the C++ TRexSelector library.

#### Package infrastructure

- `cmake --install` produces a fully relocatable install tree:
  `include/`, `lib/`, `lib/cmake/TRexSelector/`.
- Downstream projects can use `find_package(TRexSelector REQUIRED)` and link
  against `TRexSelector::trex_selector_methods`.

#### CMake changes

- All four library targets declare `FILE_SET HEADERS` for correct header
  installation and IDE indexing.
- Single authoritative `install(EXPORT TRexSelectorTargets)` in root
  `CMakeLists.txt`.
- `CMakePresets.json`: added `release-libs`, `vcpkg-release` presets.

#### CPack packaging

- `cmake --build <dir> --target package` produces binary archives.
- `cmake --build <dir> --target package_source` produces source archives.

#### vcpkg dependency manifest

- `vcpkg.json` declares eigen3, boost-headers, boost-iostreams, cereal, and
  libomp (macOS only) as managed dependencies.

#### Licensing

- GNU General Public License v3 added.
