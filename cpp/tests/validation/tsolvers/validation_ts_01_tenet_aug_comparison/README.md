# Validation: TENET vs. TENETAug Equivalence

## Purpose

Equivalence check across three elastic-net computations on the same data:

- **`TENET_Solver`**: Gram-matrix-based elastic net
- **`TENETAug_Solver`**: augmented-LASSO formulation of elastic net (data augmentation trick)
- **manual `TLASSO` on hand-augmented matrices**: a literal replica of the R `lasso_star` reference (plain LASSO on the augmented `Xstar`/`ystar` system, then de-normalised by `1/d2`)

All three are pre-scaled *externally* before the solvers run (each solver is constructed with `normalize=false`, `intercept=false`), and the check is repeated under two scaling scenarios:

1. **Z-score** scaling (centre + Bessel-corrected sample standard deviation), and
2. **L2-norm** scaling (centre + column L2 norm).

The goal is to confirm the Gram-based and augmented-LASSO elastic-net paths agree with each other and with the R `lasso_star` reference.

---

## Cross-language legs (all co-located here)

| File | Role |
| ---- | ---- |
| `validation_ts_01_tenet_aug_comparison.cpp` | C++ diagnostic (self-generating; always exits 0) |
| `validation_ts_01_tenet_aug_comparison.R` | R6-binding port of the same three-way equivalence check |
| `validation_ts_01_tenet_aug_comparison.py` | Python-binding port |
| `ts_demo_utils.{R,py}` | shared synthetic-data helper, co-located so each leg is self-contained |

Each leg synthesises its own data (no reference dumps), so they just run.

---

## Running

```bash
# C++ (compile via build_diagnostics, then run the binary)
./build/debug/bin/.../validation_ts_01_tenet_aug_comparison

# R6 bindings
Rscript validation_ts_01_tenet_aug_comparison.R

# Python bindings
python validation_ts_01_tenet_aug_comparison.py
```

---

## What to Look For

- Pairwise L2 agreement (within numerical tolerance) among the three paths: `TENET`, `TENETAug`, and the manual `lasso_star` replica
- Per-step RMSE to the true beta (de-normalised back to the original data scale)
- Consistency across both the Z-score and L2-norm scaling scenarios

---

**Last updated**: 2026-07-08
