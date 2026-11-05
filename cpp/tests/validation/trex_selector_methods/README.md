# T-Rex selector diagnostics (migrated from the examples package)

These are **investigative** probes/dumpers/comparisons, not pass/fail gates: they
print numbers for a human to read and always exit 0. They were migrated from
`TRexSelector_Examples/cpp/trex_selector_methods/validation/` so they live next to
the library code they exercise. Each program keeps the simulation-utility header it
depends on co-located in the same folder, and — for the `trex_spca` probes — the R
reference recipes and their baked dumps are co-located too (see below).

They are **not** part of the unit-test suite and never run under `ctest`. Compile
them all with:

```bash
cmake --build build/debug --target build_diagnostics
```

Then run whichever one you need from `build/debug/bin/` (most are multi-minute
Monte-Carlo studies, so they are not auto-run).

## Programs

| Program | Folder | Notes |
| ------- | ------ | ----- |
| `validation_trex_01_scaling_comparison` | `trex/` | Base T-Rex L2 vs z-score scaling invariance |
| `validation_trex_gvs_01_scaling_solver_comparison` | `trex_gvs/` | GVS/EN scaling probe — **throws at runtime** by design: it exercises the `gvs_type=EN` + `TLARS` combo the library now rejects (the GVS-EN FDR fix). Kept compile-only. |
| `validation_trex_da_01_bt_dendro_diag` | `trex_da/` | Dependency-Aware bootstrap-dendrogram diagnostic; writes `diag_*.csv` compared by the sibling `diag_bt_dendro_compare.R` |
| `validation_trex_spca_02_solver_comparison` | `trex_spca/` | TENET vs TENET_AUG equivalence |
| `validation_trex_spca_03_scaling_comparison` | `trex_spca/` | Unit-L2 vs z-score scaling |
| `validation_trex_spca_04_rdump_pipeline` | `trex_spca/` | Step-by-step C++/R diff; reads the baked `rdump/` (injected via `TREX_SPCA_RDUMP_DIR`; `--dir` overrides) |
| `validation_trex_spca_05_lambda2_foldmatch` | `trex_spca/` | Identical-fold CV curve comparison; reads committed `lambda2_probe_*` / `fm_*` refs |
| `validation_trex_spca_06_handrolled_comparison` | `trex_spca/` | Hand-rolled legacy-R recipe vs the `TRexSPCA` class |
| `validation_trex_spca_07_matrix_dump` | `trex_spca/` | Dumps GVS Phi/FDP calibration matrices; reads the baked `rdump10/` (injected via `TREX_SPCA_RDUMP10_DIR`; `--dir` overrides) |

### `trex_spca/` R reference material (co-located)

The R side of the SPCA checks lives beside the C++ probes:

| File | Role |
| ---- | ---- |
| `demo_trex_spca_02.R` | generator for `rdump/` (X designs, `truth`, `r_lambda2`, `r_pc1`, `r_results*`); run with `RDUMP_DIR=rdump10` to (re)build `rdump10/` |
| `lambda2_probe.R`, `lambda2_foldmatch.R` | produce the `lambda2_probe_*` / `fm_*` references in `validation_results/` (read by spca_05) |
| `probe_R_dummy_variance.R` | dummy-variance sensitivity probe over the baked `rdump10/` |
| `rdump/`, `rdump10/` | baked R dumps (~12 MB each) that spca_04 / spca_07 read by default |

A former `validation_trex_spca_01_lambda2_probe` was **not migrated**: it relied on
removed CV API (`ridge_cv_glmnet` / `elastic_net_cv_gaussian`). Its committed outputs
`lambda2_probe_X.csv` / `lambda2_probe_y.csv` are retained beside program 05, which
consumes them.
