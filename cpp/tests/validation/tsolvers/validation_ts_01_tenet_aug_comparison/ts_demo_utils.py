# ==============================================================================
# ts_demo_utils.py
# ==============================================================================
#
# Shared helpers for the standalone terminating-solver (tsolvers) demos.
#
# Mirrors the C++ utilities used by cpp/tsolvers/demo_ts_*:
#   - datagen::SyntheticData with predictor_policy::Normal(),
#     dummygen::Distribution::Normal(), noisegen::noise_policy::Normal()
#   - cdiagnostics::print_section_header / print_talgo_demo_config /
#     print_selection / print_selection_quality
#
# Public functions:
#   gen_synthetic_data(n, p, num_dummies, support, coefs, snr, seed)
#   print_section_header(title)
#   print_demo_config(n, p, num_dummies, T_stop, support, coefs, snr)
#   print_selection(solver, true_support)
#   print_selection_quality(solver, true_support)
#
# ==============================================================================

import numpy as np

from trex_selector_neo.utils import (
    compute_fdp,
    compute_precision,
    compute_recall,
    compute_tpp,
    f1_score,
)


def gen_synthetic_data(n, p, num_dummies, support, coefs, snr=1.0, seed=None):
    """
    Synthetic regression data with explicit dummy matrix, mirroring
    C++ datagen::SyntheticData.

        X[i, j]  i.i.d. N(0, 1)        (n x p signal predictors)
        D[i, j]  i.i.d. N(0, 1)        (n x num_dummies dummies)
        y = X[:, support] @ coefs + eps,   eps ~ N(0, noise_sigma^2 I_n)

    The noise standard deviation is SNR-calibrated:

        noise_sigma = sqrt( Var(signal) / snr )

    with the same small-sample rule as C++ noisegen::calculate_noise_params():
    Bessel-corrected variance (ddof=1) for n <= 100, population variance
    (ddof=0) otherwise.

    Parameters
    ----------
    n, p, num_dummies : int
        Observations, signal predictors, dummy predictors.
    support : array-like of int
        0-based indices of active predictors (order pairs with `coefs`).
    coefs : array-like of float
        Active coefficients, positionally matched to `support`.
    snr : float
        Target signal-to-noise ratio. Default 1.0.
    seed : int or None
        Seed for np.random.default_rng (governs X, D, and noise).

    Returns
    -------
    (X, D, y) : Fortran-ordered float64 arrays of shapes
                (n, p), (n, num_dummies), (n,).
    """
    support = np.asarray(support, dtype=np.intp)
    coefs = np.asarray(coefs, dtype=np.float64)
    assert support.shape == coefs.shape, "support/coefs size mismatch"
    assert np.all(support >= 0) and np.all(support < p), "support out of range"

    rng = np.random.default_rng(seed)
    X = np.asfortranarray(rng.standard_normal((n, p)))
    D = np.asfortranarray(rng.standard_normal((n, num_dummies)))

    signal = X[:, support] @ coefs
    ddof = 1 if n <= 100 else 0
    var_sig = np.var(signal, ddof=ddof)
    noise_sigma = np.sqrt(var_sig / snr) if (snr > 0 and var_sig > 0) else 0.0
    y = signal + rng.standard_normal(n) * noise_sigma

    return X, D, y


def print_section_header(title):
    """Mirrors cdiagnostics::print_section_header."""
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def print_demo_config(n, p, num_dummies, T_stop, support, coefs, snr):
    """Mirrors cdiagnostics::print_talgo_demo_config."""
    print("Configuration:")
    print(f"  n            = {n}")
    print(f"  p            = {p}")
    print(f"  num_dummies  = {num_dummies}")
    print(f"  T_stop       = {T_stop}")
    print(f"  true_support = {list(support)}   (0-based)")
    print(f"  true_coefs   = {list(coefs)}")
    print(f"  snr          = {snr}")


def print_selection(solver, true_support):
    """
    Mirrors cdiagnostics::print_selection: selected predictors (0-based),
    active dummy count, and number of path steps.
    """
    selected = sorted(solver.getActivePredictorIndices())
    n_dummies = len(solver.getActiveDummyIndices())
    hits = sorted(set(selected) & set(int(i) for i in true_support))
    print("\nSelection:")
    print(f"  Steps taken        : {solver.getNumSteps()}")
    print(f"  Selected predictors: {selected}")
    print(f"  True support hits  : {hits}")
    print(f"  Active dummies     : {n_dummies}")


def print_selection_quality(solver, true_support):
    """
    Mirrors cdiagnostics::print_selection_quality: TPP/FDP, precision,
    recall, and F1 of the selected predictor set vs. the true support.
    """
    selected = solver.getActivePredictorIndices()
    true_support = [int(i) for i in true_support]
    tpp = compute_tpp(selected, true_support)
    fdp = compute_fdp(selected, true_support)
    prec = compute_precision(selected, true_support)
    rec = compute_recall(selected, true_support)
    print("Selection quality:")
    print(f"  TPP (recall) = {tpp:.4f}")
    print(f"  FDP          = {fdp:.4f}")
    print(f"  Precision    = {prec:.4f}")
    print(f"  Recall       = {rec:.4f}")
    print(f"  F1 score     = {f1_score(prec, rec):.4f}")
