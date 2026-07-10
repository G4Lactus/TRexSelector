# Validation: T-LARS/T-LASSO vs. R `tlars` Path Equivalence

## Purpose

Final forward-selection path-equivalence test comparing the CRAN `tlars` package against the C++ `TLARS`/`TLASSO` solvers, using the same sparse factor-model data as the SPCA demos (`trex_spca`).

This is the authoritative cross-language correctness check for the two most commonly used solvers in the T-Rex selector.

---

## Cross-language legs (all co-located here)

| File | Role |
| ---- | ---- |
| `validation_ts_02_tlars_tlasso_rcompare.cpp` | C++ gate (exit 0/1) — run via `run_validations` |
| `demo_ts_compare_tlars_tlasso.R` | CRAN-`tlars` reference **generator** |
| `validation_ts_02_tlars_tlasso_rcompare.py` | Python-binding mirror of the C++ gate |

The C++ and Python programs do **not** generate their own data: they read the
R-produced reference CSVs (`meta.csv`, `Xn_<i>.csv`, `Dn_<i>.csv`, `y_<i>.csv`, and
the `r_lar_*`, `r_lasso_*`, `r_en_*`, `r_enlar_*` beta/action files) committed in the
shared reference folder [`../../data/rdump_tlars/`](../../data/rdump_tlars/). The R
generator writes there directly (two levels up), so re-running it refreshes the frozen
dumps in place.

The generator deliberately uses CRAN `tlars` (and `TRexSelector` internals for the GVS
dummy-from-clustering reference) as an **independent external ground truth** — it is not
legacy code awaiting migration; replacing it with the bindings it validates would defeat
the check.

---

## Running

```bash
# C++ gate — reads ../../data/rdump_tlars by default (TREX_VALIDATION_RDUMP_DIR);
# --dir <path> overrides.
cmake --build build/debug --target run_validations   # or run the binary directly

# Python mirror — same default reference dir, same --dir override.
python validation_ts_02_tlars_tlasso_rcompare.py

# Regenerate the frozen reference dumps (needs CRAN tlars + TRexSelector):
Rscript demo_ts_compare_tlars_tlasso.R
```

---

## What to Look For

- Exact (or near machine-precision) agreement between C++/Python and R selection paths and coefficients
- Any divergence points (e.g., tie-breaking differences) should be flagged and documented

---

**Last updated**: 2026-07-08
