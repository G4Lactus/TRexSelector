# Validation reference data (frozen R dumps)

## What this is

`rdump_tlars/` holds a **frozen** set of R-generated reference outputs used by the
opt-in cross-language validations in `tests/validation/`:

- `validation_mlm_hac_01_rcompare` — C++ single-linkage HAC vs R `hclust(method="single")`.
- `validation_mlm_hac_02_gvs_dummy_rcompare` — C++ GVS dummy clustering/covariance vs
  R `TRexSelector:::add_dummies_GVS()`.
- `validation_ts_02_tlars_tlasso_rcompare` — C++ TLARS / TLASSO / TENET / TENET_AUG
  solvers vs R `lars` / lasso / elastic-net reference paths.

The files are committed to the repository so the validations are self-contained: running
them needs no R toolchain and no access to any sibling repository — only the CSVs here.

## Which files are included

The **full** dump is committed, since the three validations together read all of it:

- `meta.csv`, `gvs_corrmax.csv`
- `Xn_<i>.csv` (design matrices)
- `r_clust_height_<i>.csv`, `r_clust_labels_<i>.csv` (HAC validation 01)
- `r_gvs_labels_<i>_<tag>.csv`, `r_gvs_sigma_<i>_<tag>.csv` (GVS validation 02)
- `Dn_<i>.csv`, `y_<i>.csv` (t-solver inputs, ts_02)
- `r_lar_*`, `r_lasso_*`, `r_en_*`, `r_enlar_*` (t-solver reference paths, ts_02)

## Provenance / how to regenerate

These CSVs are produced by the R generator co-located with the t-solver gate:

```
tests/validation/tsolvers/validation_ts_02_tlars_tlasso_rcompare/demo_ts_compare_tlars_tlasso.R
```

It writes directly into this folder (its `outdir` resolves to `../../data/rdump_tlars`),
so refreshing the reference data (only needed if the reference algorithmic behavior
legitimately changes) is just re-running that script — no copy step:

```bash
Rscript tests/validation/tsolvers/validation_ts_02_tlars_tlasso_rcompare/demo_ts_compare_tlars_tlasso.R
```

It requires the CRAN `tlars` and `TRexSelector` packages (the independent external
ground truth).

## How the validations find it

`tests/validation/CMakeLists.txt` injects this directory's path into each program at
compile time via `TREX_VALIDATION_RDUMP_DIR`. Each program also accepts `--dir <path>` to
point at a different dump.
