# ==============================================================================
# validation_ts_01_tenet_aug_comparison.py
# ==============================================================================
#
# Equivalence check between the Gram-based TENET_Solver, the augmented-LASSO
# TENETAug_Solver, and a manual elastic-net-to-LASSO augmentation solved with
# TLASSO_Solver (the R "lasso_star" recipe).
#
# Mirrors the co-located validation_ts_01_tenet_aug_comparison.cpp.
#
# Two scaling scenarios are checked on the same dataset:
#   Scenario 1: Z-score scaling (center + Bessel SD)
#   Scenario 2: L2-norm scaling (center + unit-L2 columns)
#
# All three solvers run their COMPLETE path (early_stop=False); the beta paths
# are compared step by step in normalized space, and de-normalized back to the
# original scale for RMSE against the known true coefficients.
#
# ==============================================================================

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# ts_demo_utils.py is co-located in this folder.
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
from trex_selector_neo.ml_methods import LpNormScaler, NormType, ZScoreScaler
from trex_selector_neo.tsolvers.lars_based import (
    TENET_Solver,
    TENETAug_Solver,
    TLASSO_Solver,
)

from ts_demo_utils import gen_synthetic_data


# ==============================================================================
# Comparison for one scaling scenario
# ==============================================================================

def run_comparison(scenario_name, X, D, y, true_beta_X, scaler_type,
                   lambda2, T_stop, p):
    n = X.shape[0]
    L = D.shape[1]
    print(f"\n{'=' * 78}\n{scenario_name}\n{'=' * 78}")

    # Per-column scale of the RAW X (before scaling), used to de-normalize
    # the coefficient paths back to the original scale:
    #   ZScore -> Bessel-corrected sample SD, L2 -> centered column L2 norm.
    scale_div = np.sqrt(n - 1) if scaler_type == "zscore" else 1.0
    Xc = X - X.mean(axis=0, keepdims=True)
    scale_X = np.linalg.norm(Xc, axis=0) / scale_div

    # Apply the standardizer in place (separate instances for X and D).
    if scaler_type == "zscore":
        sc_X = ZScoreScaler(center=True, scale=True)
        sc_D = ZScoreScaler(center=True, scale=True)
    else:
        sc_X = LpNormScaler(NormType.L2, center=True)
        sc_D = LpNormScaler(NormType.L2, center=True)
    sc_X.fit(X)
    sc_X.transform_inplace(X)
    sc_D.fit(D)
    sc_D.transform_inplace(D)
    y -= y.mean()

    # TENET: Gram-based elastic net, full path.
    tenet = TENET_Solver(X, D, y, lambda2, normalize=False, intercept=False,
                         verbose=False)
    tenet.executeStep(T_stop, early_stop=False)

    # TENETAug: augmented-LASSO elastic net, full path.
    tenet_aug = TENETAug_Solver(X, D, y, lambda2, normalize=False,
                                intercept=False, verbose=False)
    tenet_aug.executeStep(T_stop, early_stop=False)

    # Manual TLASSO on hand-augmented matrices (the R lasso_star recipe):
    #   d1 = sqrt(lambda2), d2 = 1/sqrt(1+lambda2)
    #   X_aug = [d2*X; d2*d1*I_p (rows n..n+p-1)],  D_aug analogous below it,
    #   y_aug = [y; 0_{p+L}],  beta_original = beta_star / d2.
    d1 = np.sqrt(lambda2)
    d2 = 1.0 / np.sqrt(1.0 + lambda2)
    naug = n + p + L
    X_aug = np.zeros((naug, p), order="F")
    X_aug[:n, :] = d2 * X
    X_aug[n:n + p, :] = d2 * d1 * np.eye(p)
    D_aug = np.zeros((naug, L), order="F")
    D_aug[:n, :] = d2 * D
    D_aug[n + p:, :] = d2 * d1 * np.eye(L)
    y_aug = np.concatenate([y, np.zeros(p + L)])

    tlasso_aug = TLASSO_Solver(X_aug, D_aug, y_aug, normalize=False,
                               intercept=False, verbose=False)
    tlasso_aug.executeStep(T_stop, early_stop=False)

    # Beta paths: (p+L) x (steps+1); manual path de-normalized by 1/d2.
    path_tenet = tenet.getBetaPath()
    path_aug = tenet_aug.getBetaPath()
    path_manual = tlasso_aug.getBetaPath() / d2
    n_steps = min(path_tenet.shape[1], path_aug.shape[1], path_manual.shape[1])
    print(f"Path lengths: TENET = {path_tenet.shape[1]}, "
          f"TENET_AUG = {path_aug.shape[1]}, TLASSO_aug = {path_manual.shape[1]}")

    inv_sqrt_p = 1.0 / np.sqrt(p)
    rmse = {k: [] for k in ("ten", "aug", "man")}
    dists = {k: [] for k in ("ten_aug", "ten_man", "aug_man")}
    for k in range(n_steps):
        bx = {
            "ten": path_tenet[:p, k],
            "aug": path_aug[:p, k],
            "man": path_manual[:p, k],
        }
        for key, b in bx.items():
            o = b / scale_X
            rmse[key].append(np.linalg.norm(o - true_beta_X) * inv_sqrt_p)
        dists["ten_aug"].append(np.linalg.norm(bx["ten"] - bx["aug"]))
        dists["ten_man"].append(np.linalg.norm(bx["ten"] - bx["man"]))
        dists["aug_man"].append(np.linalg.norm(bx["aug"] - bx["man"]))

    # Sparse per-step table: every 5th step plus the last.
    w = 13
    hdr = ["step", "RMSE_TEN", "RMSE_AUG", "RMSE_TLS",
           "d(TEN,AUG)", "d(TEN,TLS)", "d(AUG,TLS)"]
    print("".join(f"{h:>{w}}" for h in hdr))
    for k in range(n_steps):
        if k % 5 != 0 and k != n_steps - 1:
            continue
        row = (f"{k:>{w}}"
               + f"{rmse['ten'][k]:>{w}.5f}"
               + f"{rmse['aug'][k]:>{w}.5f}"
               + f"{rmse['man'][k]:>{w}.5f}"
               + f"{dists['ten_aug'][k]:>{w}.2e}"
               + f"{dists['ten_man'][k]:>{w}.2e}"
               + f"{dists['aug_man'][k]:>{w}.2e}")
        print(row)

    # Summary.
    print("\nSummary:")
    for key, label in (("ten_aug", "TENET     vs TENET_AUG "),
                       ("ten_man", "TENET     vs TLASSO_aug"),
                       ("aug_man", "TENET_AUG vs TLASSO_aug")):
        arr = np.asarray(dists[key])
        print(f"  {label}: max L2 = {arr.max():.4e}, mean L2 = {arr.mean():.4e}")
    for key, label in (("ten", "TENET     "), ("aug", "TENET_AUG "),
                       ("man", "TLASSO_aug")):
        arr = np.asarray(rmse[key])
        print(f"  {label} RMSE: best = {arr.min():.6f}, final = {arr[-1]:.6f}")

    # Verdicts.
    def verdict(max_d):
        if max_d < 1e-8:
            return "EQUIVALENT"
        if max_d < 1e-4:
            return "NEARLY EQUIVALENT"
        return f"DIFFER (max L2 = {max_d:.4e})"

    print("\nVerdicts:")
    print(f"  TENET     vs TENET_AUG : {verdict(max(dists['ten_aug']))}")
    print(f"  TENET     vs TLASSO_aug: {verdict(max(dists['ten_man']))}")
    print(f"  TENET_AUG vs TLASSO_aug: {verdict(max(dists['aug_man']))}")

    return max(max(dists["ten_aug"]), max(dists["ten_man"]),
               max(dists["aug_man"]))


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    n, p = 90, 150
    L = 3 * p
    lambda2 = 100.01
    T_stop = 10          # unused: early_stop=False traces the complete path
    snr = 1.0

    true_support = [10, 50, 85]
    true_coefs = [2.5, -1.8, 3.2]
    true_beta_X = np.zeros(p)
    true_beta_X[true_support] = true_coefs

    print(f"Generating data: n={n} p={p} L={L} lambda2={lambda2}")
    X_raw, D_raw, y_raw = gen_synthetic_data(n, p, L, true_support, true_coefs,
                                             snr, seed=42)

    worst = 0.0
    for name, sc in (
        ("Scenario 1: Z-score scaling (centre + SD)", "zscore"),
        ("Scenario 2: L2-norm scaling (centre + L2-normalise)", "l2"),
    ):
        # Fresh copies per scenario (scaling mutates in place).
        worst = max(worst, run_comparison(
            name, np.asfortranarray(X_raw.copy()),
            np.asfortranarray(D_raw.copy()), y_raw.copy(),
            true_beta_X, sc, lambda2, T_stop, p))

    print(f"\nOverall worst pairwise max L2 = {worst:.4e}")
