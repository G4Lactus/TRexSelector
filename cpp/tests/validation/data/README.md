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

These CSVs were produced by the R generator in the examples repository:

```
TRexSelector_Examples/R/tsolvers/validation/validation_ts_02_tlars_tlasso_rcompare/demo_ts_compare_tlars_tlasso.R
```

which writes an `rdump_tlars/` folder next to itself. To refresh this reference data (only
needed if the reference algorithmic behavior legitimately changes), re-run that script and
copy the subset listed above back into this folder.

## How the validations find it

`tests/validation/CMakeLists.txt` injects this directory's path into each program at
compile time via `TREX_VALIDATION_RDUMP_DIR`. Each program also accepts `--dir <path>` to
point at a different dump.
