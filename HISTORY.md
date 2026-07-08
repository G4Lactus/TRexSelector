# TRexSelector changelog

## TRexSelector ?.?.?

????-??-??

### 2026-07-08

#### R <-> Python binding parity

- BREAKING (R): `TRexBiobankScreeningSelector$select()` now returns the same
  per-phenotype record shape as the Python binding. A matrix `Y` previously
  returned a list with a `$statistics` data.frame plus parallel top-level
  `$selected_indices` / `$selected_indices_screen_ordinary` /
  `$selected_indices_screen_bootstrap` lists; it now returns a plain list of
  records, one per phenotype, where `res[[i]]` has the identical shape to the
  single-phenotype (vector `Y`) return. No information is lost — every
  `$statistics` column and index list is a field of each record — and the
  summary table is a one-line projection:
  `do.call(rbind, lapply(res, function(r) as.data.frame(r[c("phenotype_index", "estimated_FDR", "method_used", "used_fallback_trex")])))`.
  This matches Python, which returns one `BiobankScreenTRexResult` (vector) or a
  list of them (matrix).
- R gains `prior_groups` + `rho_grid_labels` in `trex_da_control()` — the
  prior-groups dependency-aware path (`setupDA_PriorGroups`), which Python
  already exposed; the R glue previously could not route to it. `hc_grid_length`
  is now actually read by the DA glue (it was silently inert before).
- R gains `cv_seed` / `cv_n_folds` / `cv_n_lambda` in `trex_gvs_control()` — the
  cross-validation fold controls for the CV `lambda2_method` paths (`cv_seed`
  enables per-trial fold reproducibility), at parity with Python.
- `TRexGVSSelector` gains read-only `$groups` (1-based per-variable cluster
  labels, length `p`) and `$group_labels`, so cluster diagnostics (e.g. purity)
  can be computed R-side; Python exposes the same via `GVSSelectionResult`.

### 2026-07-07

#### Shared-buffer safety guard (new runtime check)

- `TRexSelector` (and every derived selector) normalizes the input matrix `X`
  in place via an `Eigen::Map` and restores it when `select()` finishes or on
  destruction. Two selectors mapped onto the SAME buffer therefore silently
  corrupted each other: the first to restore left the buffer un-normalized, so
  the second ran on raw data (typically selecting everything). The core now
  claims a buffer when it normalizes it and throws a clear error if a second
  live selector is constructed on the same matrix. Fix each caller by giving
  each selector its own copy of `X`, or by running selectors one at a time
  (construct, `select()`, and let each go out of scope before the next). The
  guard is in the C++ core, so C++, R, and Python all benefit; internal
  sequential orchestrators (SPCA per-PC, biobank screen->fallback) are
  unaffected because `select()` releases the claim before the next selector
  is built.

#### R <-> Python binding parity

- Elastic-net model selection is now exposed in BOTH languages: `ElasticNet`
  (glmnet-style coordinate-descent coefficient path, backed by `enet_gaussian`)
  and `ElasticNetCV` (K-fold CV lambda.min/1se selection, backed by
  `enet_cv_ccd`). `RidgeCV` (SVD-path ridge CV, `ridge_cv_svd`) is added to
  Python, which previously exposed no model-selection CV at all. Verified
  R and Python produce bit-identical elastic-net paths on shared data.
- R gains `da_method = "NN"` in `trex_da_control()` — the nearest-neighbour
  correlation-threshold DA sweep (`DAMethod::NN`), which Python already
  exposed; the R glue previously rejected it. `hc_grid_length = 0` lets the
  core auto-select `min(20, p)`.
- R gains `tenet_aug_use_lars` in `trex_gvs_control()` and
  `trex_spca_control()` — the LARS-vs-LASSO inner-path switch for `TENET_AUG`
  (Python already had it). Documentation clarifies that `en_solver`
  (`TENET` vs `TENET_AUG`) is a solver-EQUIVALENCE knob (both select the same
  variables), NOT the LARS-vs-LASSO distinction.
- `TENETAug_Solver` / `TIENETAug_Solver` are exported from the R package (were
  internal-only), with plain and mmap creators, 1-based group accessors, and
  save/load.

#### GVS / SPCA / clustering fixes

- BREAKING (R): `gvs_type = "IEN"` now derives the `TIENET_AUG` solver
  (matching the 2026-07-05 solver rework and the C++/Python wrappers); the R
  wrapper previously forced the old `TLASSO` solver.
- `TRexGVSSelector` now emits a warning when `control$solver` / `solver_type`
  is set to a value that does not match the solver derived from `gvs_type`
  (the field is silently overridden).
- `lambda_2 == 0` (degenerate no-ridge / pure T-LASSO) now emits a warning —
  the `-1` auto / `0` degenerate / `> 0` fixed sentinel convention had tripped
  in-house callers.
- Ward + `Correlation_LSH_Approx` is now permitted in both the R and Python
  clustering glues (the core dispatcher supports it, running Ward in a 64-D
  SimHash-projected Euclidean space; the glues wrongly rejected every
  non-Euclidean metric under Ward).

#### Python API

- `TRexBiobankScreeningSelector`: the control parameter `biosctrex_ctrl` is
  renamed to `bio_ctrl`, and `BiobankScreenTRexControl` now exposes its nested
  `trex_screen_ctrl` (and that control's nested `trex_ctrl`), so Python can set
  K, solver, screening method, etc., at parity with the R knobs.
- Scalers gain returning, layout-agnostic `transform()` / `fit_transform()` /
  `inverse_transform()` methods alongside the strict zero-copy
  `*_inplace` variants (which require a writeable Fortran-ordered float64
  buffer), removing the sharpest edge in the Python scaler API.

#### Core fixes

- `TENETAug_Solver` / `TIENETAug_Solver` deserialization: loading into an
  already-constructed solver (the R/Python bindings' `load()` path) now rebuilds
  the augmented maps and reconnects the inner solver unconditionally on load,
  instead of gating on a null-map check that only held for fresh-object loads —
  fixing checkpoint resume through the bindings.
- `cmake --install` now prunes the package-owned header trees before copying,
  so a renamed or deleted header can no longer poison the local install tree
  (a stale header that `#include`d a since-renamed file previously broke every
  consumer until the install was cleaned by hand).

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

#### DA lifecycle refactor and NN-setup parallelisation

- `TRexDASelector` no longer overrides `select()`: the dependency-structure
  setup moved into an `onSelectBegin()` override and the DA selection /
  result widening into a `finalizeSelectionResult()` override, so DA now
  runs the inherited base pipeline like GVS does (behavior-neutral; the
  duplicated scaffolding — voting-grid setup, loop drivers, cleanup,
  X denormalization — existed twice before and could drift).
- The DA-NN neighbourhood setup is OpenMP-parallelised: each variable's
  neighbour list is built independently (race-free row-wise scan instead of
  the symmetric pair loop), with an early exit over the ascending rho grid.
- `corr_max` (GVS) documents that the default 0.5 matches the R reference
  and that the EUSIPCO 2022 paper's GWAS experiments used rho_thr = 1/3.

#### T-Rex+GVS rework: solver-encapsulated augmentation and the TIENETAug solver

- New `TIENETAug_Solver` (tsolvers, LARS family): the Informed Elastic Net
  (Machkour et al., CAMSAP 2023, Theorem 1) as a terminating forward selector.
  Builds the group-informed row-augmented system `[X; B]`, `[D; B tiled per
  dummy layer]`, `[y; 0_M]` (B(m, j) = sqrt(lambda2)/sqrt(p_m) on group-m
  columns) internally from a 0-based group assignment, applies the R-faithful
  post-augmentation column scaling, and runs an inner T-LASSO (or pure T-LARS)
  — full save/load support, registered in the polymorphic solver tests.
- `TRexGVSSelector` now hands every solver the plain (n x p) design, the plain
  (n x L) cluster-MVN dummy block, and the centered response; all augmentation
  lives in the solvers. `GVSType::IEN` routes through `TIENETAug_Solver`
  (new `SolverTypeForTRex::TIENET_AUG`, also in Python; IEN previously
  required TLASSO — breaking config change), and `en_solver = TENET_AUG`
  routes through `TENETAug_Solver`, which finally makes the
  `tenet_aug_use_lars` R-parity flag functional (it was silently ignored).
  The inline augmentation machinery (X_aug_/y_aug_ members, five build
  helpers, the variable effective row count, and the EN-aug mmap
  reinterpretation) is deleted; memory-mapped dummy files now always have n
  rows. EN-aug thereby switches from post-augmentation re-normalization to
  the demo-verified as-constructed convention that reproduces the Gram-based
  TENET path exactly (asserted by a new equivalence test).
- Review fixes: the GVS solver cache is cleared BEFORE the dummy buffers it
  views are reassigned (transient-dangling-view hardening, mirroring the
  base-class contract); EN dummy normalization goes through the shared
  data-normalizer (degenerate columns warn and keep scale 1.0 instead of
  throwing).
- New end-to-end GVS tests (grouped-recovery for EN/TENET, EN/TENETAug,
  EN/TENETAug with pure-LARS inner, IEN/TIENETAug, plus TENET==TENETAug
  selection equivalence). The IEN test pins lambda_2: the ridge-CV value
  (identical to the R reference's, which shares one CV and one * p/2
  conversion between EN and IEN) can make the sqrt(lambda2)-scaled group
  rows dominate the augmented geometry on n > p designs, leaving nothing
  selectable — a documented sensitivity of the IEN track (see
  computeLambda2()).

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
- `tloop_stagnation_stop` is now tri-state (`std::optional<bool>`, Python:
  `bool | None`). The default (auto) resolves by solver family: DISABLED for
  the equiangular LARS-path solvers (TLARS/TLASSO/TENET/TENET_AUG), which
  terminate properly on their own — matching the R reference and the T-Rex
  paper, neither of which has a stagnation stop — and ENABLED for greedy
  solvers, which otherwise run into a noise trap and select until the
  maximum number of dummies is reached. An explicit true/false overrides the
  family default.
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
