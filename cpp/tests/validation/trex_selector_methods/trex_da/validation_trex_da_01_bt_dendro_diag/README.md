# Validation: BT Dendrogram Diagnostic (C++/R Correctness Comparison)

## Purpose

A **single-dataset correctness diagnostic**, not a Monte Carlo performance study. This validation freezes one fixed AR(1)-block dataset (the base configuration of `trex_da` Demo 04) and exports every intermediate quantity of the `BT` clustering pipeline to CSV, so that the companion R script `diag_bt_dendro_compare.R` can reproduce the *identical* pipeline in R and diff the results — isolating whether any C++/R discrepancy originates in the **clustering step** (dendrogram heights/groups) or **downstream** (voting/selection given matching clusters).

---

## Fixed Dataset

Identical to Demo 04's base configuration: `dgp_ar1_block` with $n=150$, $M=5$ blocks, $Q=5$ (block size, $p=25$), $\rho=0.7$, amplitude $1.0$, $\mathrm{SNR}=2.0$, $\mathrm{tFDR}=0.2$, $K=20$, deterministic seed $2026$ — a single fixed trial, not repeated across a grid or averaged over MC runs.

---

## What Gets Exported

1. **The design matrix $X$, response $y$, and true support** → CSV, so the R script clusters the *exact same* $X$.
2. **Single-linkage dendrogram merge heights** (correlation distance), computed with the same clustering call the selector uses internally.
3. **The $\rho$-grid** (built exactly as `TRexDASelector::setupDA_BT` does internally) and, for every $\rho$ level, the per-variable group sizes/memberships after cutting the tree at height $=1-\rho$.
4. **The full `TRexDASelector(BT, Single)` result**: $\rho$-grid, $T_{\text{stop}}$, dummy multiplier $L$, selected variables, FDP/TPP, and the chosen $\rho$ threshold.

The columns are centered and L2-normalized before computing correlation distance (`center_l2()`), so that the distance metric matches exactly what the selector's internal clustering sees (dividing out column norms reproduces $|{\text{Pearson correlation}}|$ exactly).

---

## Companion R Script (`diag_bt_dendro_compare.R`)

Reproduces the same pipeline in R on the identical exported dataset:
1. Computes the correlation matrix and converts it to a distance ($1-|\rho|$).
2. Runs single-linkage hierarchical clustering; records dendrogram merge heights.
3. Builds the same $\rho$-grid as `TRexDASelector::setupDA_BT`; cuts the dendrogram per level; records group memberships.
4. Runs T-Rex with method `"trex+DA+BT"`; records the selection result.
5. Diffs the R-side dendrogram heights and group sizes against the C++ exports, printing a PASS/FAIL summary.

---

## Output Files

Written to `validation_results/` (already populated — both C++ and R sides have been run and exported):

- `diag_X.csv`, `diag_y.csv`, `diag_support0.csv` — the frozen dataset.
- `diag_cpp_heights.csv`, `diag_cpp_groups.csv`, `diag_cpp_selector.csv` — C++-side pipeline exports.
- `diag_r_heights.csv`, `diag_r_groups.csv`, `diag_r_selector.csv` — R-side reproduction exports.

---

## Running

```bash
./build/debug/bin/trex_selector_methods/validation/trex_da/validation_trex_da_01_bt_dendro_diag/validation_trex_da_01_bt_dendro_diag
```

Then run the companion R script (from this folder, or wherever your R environment is set up) against the same `validation_results/` folder to regenerate the `diag_r_*.csv` files and print the PASS/FAIL comparison:

```r
source("diag_bt_dendro_compare.R")
```

---

## Interpretation

- This is a **debugging/verification tool**, not a benchmark — there is no FDR/TPR table to interpret; the output of interest is whether `diag_cpp_heights.csv`/`diag_cpp_groups.csv` match `diag_r_heights.csv`/`diag_r_groups.csv` (and ultimately whether `diag_cpp_selector.csv` matches `diag_r_selector.csv`).
- If dendrogram heights/groups **match** but the final selector results **diverge**, the discrepancy lies downstream of clustering (e.g. in the voting/threshold logic).
- If dendrogram heights/groups themselves **diverge**, the discrepancy lies in the clustering step (e.g. distance computation, linkage implementation, or floating-point tie-breaking differences between the C++ and R clustering routines).
- Since both the C++ and R output files already exist in `validation_results/`, you can inspect the current PASS/FAIL state directly rather than needing to rerun both sides from scratch.

---

**Last updated**: 2026-07-04
