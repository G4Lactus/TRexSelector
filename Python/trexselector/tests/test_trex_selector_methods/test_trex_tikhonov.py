"""
Tests for TRexTikhonovSelector (general-Tikhonov T-Rex).

Mirrors the R suite (test-trex_tikhonov.R): control round trip, construction
gates, the solver-family warning, a fixed-lambda_2 run with its getters, the
group-mean-K equivalence with GVS-IEN TCDIENET, the K = I collapse onto the
TCDENET run, the two K builders, and the lambda_2 tuners (CV and pilot).
"""
import warnings

import numpy as np
import pytest
import trexselector as tsn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grouped_data(seed=7, n=60, p=12, gsize=4, rho=0.7):
    """Correlated groups of size gsize; the signal sits in the first group."""
    rng = np.random.default_rng(seed)
    X = np.empty((n, p))
    for g in range(0, p, gsize):
        z = rng.standard_normal(n)
        for j in range(g, min(g + gsize, p)):
            X[:, j] = rho * z + np.sqrt(1.0 - rho * rho) * rng.standard_normal(n)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] - 2.0 * X[:, 2] + rng.standard_normal(n)
    groups = [j // gsize for j in range(p)]
    return X, y, groups


def _group_mean_k(groups):
    """W = sum_m 1_m 1_m^T / p_m (the IEN group-mean matrix)."""
    groups = np.asarray(groups)
    p = groups.size
    K = np.zeros((p, p))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        K[np.ix_(idx, idx)] = 1.0 / idx.size
    return K


def _tik_control(K=None, lambda_2=1.0, **fields):
    ctrl = tsn.TRexTikhonovControlParameter()
    if K is not None:
        ctrl.tikhonov_K = K
    ctrl.lambda_2 = lambda_2
    for name, value in fields.items():
        setattr(ctrl, name, value)
    return ctrl


def _base_control(K=5, mult=2, solver=None):
    ctrl = tsn.TRexControlParameter()
    ctrl.K = K
    ctrl.max_dummy_multiplier = mult
    if solver is not None:
        ctrl.solver_type = solver
    return ctrl


# ---------------------------------------------------------------------------
# Control parameter round trip
# ---------------------------------------------------------------------------

def test_control_fields_roundtrip():
    ctrl = tsn.TRexTikhonovControlParameter()
    # defaults mirror the C++ struct
    assert ctrl.penalty_structure == tsn.TikPenaltyStructure.SPARSE_GENERAL
    assert ctrl.lambda_2 == pytest.approx(-1.0)
    assert ctrl.lambda2_method == tsn.TikLambda2Method.PILOT_KNEE
    assert not hasattr(tsn.TikLambda2Method, "PILOT_MAX")   # the argmax variant is gone
    assert ctrl.cluster_dummies is True
    assert ctrl.fold_dummy_coupling is True
    assert ctrl.trex_ctrl.solver_type == tsn.SolverTypeForTRex.TCDTIKNET
    assert ctrl.tikhonov_K.shape == (0, 0)
    assert ctrl.cv_dfmax_ratio == pytest.approx(0.8)

    K = _group_mean_k([0, 0, 1, 1])
    ctrl.tikhonov_K = K
    np.testing.assert_allclose(ctrl.tikhonov_K, K)   # dense round trip through sparse

    ctrl.penalty_structure = tsn.TikPenaltyStructure.GROUP_CENTERING
    ctrl.lambda2_method = tsn.TikLambda2Method.PILOT_KNEE
    ctrl.pilot_statistic = tsn.PilotStatistic.GROUP_MAJORITY
    ctrl.cv_n_folds = 5
    ctrl.cv_n_lambda1 = 15
    ctrl.cv_n_lambda2 = 25
    ctrl.cv_seed = 3
    ctrl.cv_dfmax_ratio = 0.95
    ctrl.pilot_grid = [0.0, 1.0, 10.0]
    ctrl.pilot_K = 4
    ctrl.pilot_T = 2
    ctrl.pilot_L_multiplier = 1
    ctrl.pilot_knee_tolerance = 2.0
    ctrl.pilot_group_fraction = 0.6
    ctrl.pilot_shared_dummies = True
    ctrl.fold_dummy_coupling = False
    ctrl.cluster_dummies = False
    ctrl.prior_groups = [0, 0, 1, 1]
    ctrl.corr_max = 0.3
    assert ctrl.penalty_structure == tsn.TikPenaltyStructure.GROUP_CENTERING
    assert ctrl.lambda2_method == tsn.TikLambda2Method.PILOT_KNEE
    assert ctrl.pilot_statistic == tsn.PilotStatistic.GROUP_MAJORITY
    assert (ctrl.cv_n_folds, ctrl.cv_n_lambda1, ctrl.cv_n_lambda2, ctrl.cv_seed) == (5, 15, 25, 3)
    assert ctrl.cv_dfmax_ratio == pytest.approx(0.95)
    assert list(ctrl.pilot_grid) == [0.0, 1.0, 10.0]
    assert (ctrl.pilot_K, ctrl.pilot_T, ctrl.pilot_L_multiplier) == (4, 2, 1)
    assert ctrl.pilot_knee_tolerance == pytest.approx(2.0)
    assert ctrl.pilot_group_fraction == pytest.approx(0.6)
    assert ctrl.pilot_shared_dummies is True
    assert ctrl.fold_dummy_coupling is False
    assert ctrl.cluster_dummies is False
    assert list(ctrl.prior_groups) == [0, 0, 1, 1]
    assert ctrl.corr_max == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Construction gates
# ---------------------------------------------------------------------------

def test_tik_control_is_required():
    X, y, _ = _grouped_data()
    with pytest.raises(ValueError, match="tik_control is required"):
        tsn.TRexTikhonovSelector(X, y, trex_control=_base_control(), verbose=False)


def test_empty_or_misdimensioned_k_raises():
    X, y, groups = _grouped_data()
    # SPARSE_GENERAL without a K
    with pytest.raises(Exception):
        tsn.TRexTikhonovSelector(X, y, tik_control=_tik_control(None, 1.0),
                                 trex_control=_base_control(), verbose=False)
    # K of the wrong size
    with pytest.raises(Exception):
        tsn.TRexTikhonovSelector(X, y, tik_control=_tik_control(np.eye(X.shape[1] - 1), 1.0),
                                 trex_control=_base_control(), verbose=False)


def test_non_family_solver_warns_and_uses_tcdtiknet():
    X, y, groups = _grouped_data()
    tik = _tik_control(_group_mean_k(groups), 1.0, prior_groups=groups)
    with pytest.warns(UserWarning, match="not a Tikhonov-family solver"):
        sel = tsn.TRexTikhonovSelector(
            X, y, tik_control=tik,
            trex_control=_base_control(solver=tsn.SolverTypeForTRex.TLASSO),
            seed=1, verbose=False)
    sel.select()
    assert sel.selected_var.shape == (X.shape[1],)


# ---------------------------------------------------------------------------
# A fixed-lambda_2 run and the getters
# ---------------------------------------------------------------------------

def test_fixed_lambda2_run_and_getters():
    X, y, groups = _grouped_data()
    tik = _tik_control(_group_mean_k(groups), 2.5, prior_groups=groups)
    sel = tsn.TRexTikhonovSelector(X, y, tik_control=tik,
                                   trex_control=_base_control(), seed=1, verbose=False)
    assert sel.lambda2_used == pytest.approx(-1.0)          # before select()
    res = sel.select()
    assert isinstance(res, tsn.SelectionResult)
    assert sel.lambda2_used == pytest.approx(2.5)
    assert sel.lambda2_cv_diagnostics.ran is False
    assert sel.lambda2_pilot.ran is False
    assert all(0 <= i < X.shape[1] for i in sel.selected_indices)
    assert sel.T_stop >= 1
    assert 1 <= sel.L <= 2


def test_group_centering_structure_runs():
    X, y, groups = _grouped_data()
    tik = _tik_control(None, 1.0, prior_groups=groups,
                       penalty_structure=tsn.TikPenaltyStructure.GROUP_CENTERING)
    sel = tsn.TRexTikhonovSelector(X, y, tik_control=tik,
                                   trex_control=_base_control(), seed=1, verbose=False)
    sel.select()
    assert sel.selected_var.shape == (X.shape[1],)


# ---------------------------------------------------------------------------
# Equivalences pinned in C++ and R
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lambda_2", [10.0, 0.0])
def test_group_mean_k_matches_gvs_ien_tcdienet(lambda_2):
    """Folded group-mean K with prior groups is the GVS-IEN TCDIENET run."""
    X, y, groups = _grouped_data(seed=11, n=80, p=16)
    K = _group_mean_k(groups)

    tik = _tik_control(K, lambda_2, prior_groups=groups)
    sel_tik = tsn.TRexTikhonovSelector(X.copy(), y.copy(), tik_control=tik,
                                       trex_control=_base_control(K=5, mult=3),
                                       seed=42, verbose=False)
    sel_tik.select()

    gvs = tsn.TRexGVSControlParameter()
    gvs.gvs_type = tsn.GVSType.IEN
    gvs.prior_groups = groups
    gvs.lambda_2 = lambda_2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # lambda_2 = 0 is deliberate
        sel_gvs = tsn.TRexGVSSelector(
            X.copy(), y.copy(), gvs_control=gvs,
            trex_control=_base_control(K=5, mult=3, solver=tsn.SolverTypeForTRex.TCDIENET),
            seed=42, verbose=False)
    sel_gvs.select()

    np.testing.assert_array_equal(sel_tik.selected_var, sel_gvs.selected_var)


def test_identity_k_with_both_escape_hatches_collapses_onto_tcdenet():
    """K = I, i.i.d. dummies, no folding: the TCDENET run at the same lambda_2."""
    X, y, _ = _grouped_data(seed=5)
    lambda_2 = 1.5

    tik = _tik_control(np.eye(X.shape[1]), lambda_2,
                       cluster_dummies=False, fold_dummy_coupling=False)
    sel_tik = tsn.TRexTikhonovSelector(X.copy(), y.copy(), tik_control=tik,
                                       trex_control=_base_control(K=5, mult=2),
                                       seed=42, verbose=False)
    sel_tik.select()

    base = _base_control(K=5, mult=2, solver=tsn.SolverTypeForTRex.TCDENET)
    base.solver_params.lambda2 = lambda_2
    sel_en = tsn.TRexSelector(X.copy(), y.copy(), trex_control=base, seed=42, verbose=False)
    sel_en.select()

    np.testing.assert_array_equal(sel_tik.selected_var, sel_en.selected_var)


# ---------------------------------------------------------------------------
# K builders
# ---------------------------------------------------------------------------

def test_gamma_to_k_builds_the_chain_laplacian():
    p = 6
    gamma = np.zeros((p - 1, p))
    for i in range(p - 1):
        gamma[i, i] = -1.0
        gamma[i, i + 1] = 1.0
    K = tsn.gamma_to_k(gamma)
    expected = gamma.T @ gamma
    np.testing.assert_allclose(K, expected)
    assert K[0, 0] == pytest.approx(1.0) and K[1, 1] == pytest.approx(2.0)
    assert K[0, 1] == pytest.approx(-1.0)


def test_build_group_centering_k():
    K = tsn.build_group_centering_k([0, 0, 1, 1, 1])
    expected = np.zeros((5, 5))
    expected[:2, :2] = np.eye(2) - np.ones((2, 2)) / 2.0
    expected[2:, 2:] = np.eye(3) - np.ones((3, 3)) / 3.0
    np.testing.assert_allclose(K, expected)


# ---------------------------------------------------------------------------
# lambda_2 resolution: CV tuner and pilot rule
# ---------------------------------------------------------------------------

def test_lambda2_resolves_via_the_ien_tuner():
    X, y, groups = _grouped_data(seed=3)
    tik = _tik_control(_group_mean_k(groups), -1.0, prior_groups=groups,
                       lambda2_method=tsn.TikLambda2Method.CV_1SE_IEN_CCD,
                       cv_n_folds=5, cv_n_lambda1=15, cv_n_lambda2=12, cv_seed=1,
                       cv_dfmax_ratio=0.9)
    sel = tsn.TRexTikhonovSelector(X, y, tik_control=tik,
                                   trex_control=_base_control(), seed=1, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # a boundary pick may warn on tiny data
        sel.select()
    diag = sel.lambda2_cv_diagnostics
    assert diag.ran is True
    assert diag.grid_size == 12
    assert 0 <= diag.grid_index < 12
    assert sel.lambda2_used >= 0.0
    assert sel.lambda2_pilot.ran is False
    # The support-cap ratio reached the tuner and is recorded as passed.
    assert diag.dfmax_ratio == pytest.approx(0.9)
    assert diag.profile_at_edge or not diag.profile_at_support_cap


def test_non_finite_cv_dfmax_ratio_raises():
    X, y, groups = _grouped_data()
    tik = _tik_control(_group_mean_k(groups), -1.0, prior_groups=groups,
                       cv_dfmax_ratio=float("nan"))
    with pytest.raises(ValueError, match="cv_dfmax_ratio must be finite"):
        tsn.TRexTikhonovSelector(X, y, tik_control=tik,
                                 trex_control=_base_control(), seed=1, verbose=False)


def test_lambda2_resolves_via_the_pilot_rule():
    X, y, groups = _grouped_data(seed=3)
    grid = [0.0, 1.0, 10.0, 100.0]
    tik = _tik_control(_group_mean_k(groups), -1.0, prior_groups=groups,
                       lambda2_method=tsn.TikLambda2Method.PILOT_KNEE,
                       pilot_grid=grid, pilot_K=3, pilot_T=2, pilot_L_multiplier=1)
    sel = tsn.TRexTikhonovSelector(X, y, tik_control=tik,
                                   trex_control=_base_control(), seed=1, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel.select()
    pilot = sel.lambda2_pilot
    assert pilot.ran is True
    assert list(pilot.grid) == grid
    assert len(pilot.counts) == len(grid)
    assert pilot.lambda2 in grid
    assert pilot.index == grid.index(pilot.lambda2)
    assert pilot.pick.max_selected >= 0
    # L_0 is the pilot's dummy count: pilot_L_multiplier * p
    assert (pilot.K_pilot, pilot.T_pilot, pilot.L_0) == (3, 2, X.shape[1])
    assert sel.lambda2_used == pytest.approx(pilot.lambda2)
    # The CV record notes that a tuner ran (and its seconds); the grid lives
    # in the pilot record, so the CV grid fields stay unset.
    diag = sel.lambda2_cv_diagnostics
    assert diag.ran is True
    assert diag.grid_size == 0 and diag.grid_index == -1


# ---------------------------------------------------------------------------
# The full-profile knee (one selection per grid value)
# ---------------------------------------------------------------------------

def test_lambda2_power_rule_profiles_the_grid_and_keeps_the_contract():
    X, y, groups = _grouped_data(seed=3)
    X_before = X.copy()
    tik = _tik_control(_group_mean_k(groups), -1.0, prior_groups=groups)
    base = _base_control()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = tsn.tikhonov_lambda2_power_rule(X, y, tFDR=0.2, tik_control=tik, trex_control=base,
                                               seed=1, grid=[10.0, 0.0, 1.0, 10.0])
    grid = [p.lambda2 for p in prof.probes]
    assert grid == [0.0, 1.0, 10.0]                       # sorted, de-duplicated
    counts = list(prof.counts())
    assert counts == [len(p.selected) for p in prof.probes]
    assert prof.index == next(k for k, c in enumerate(counts) if c >= max(counts) - 1)   # the knee
    assert prof.lambda2 == grid[prof.index]
    assert list(prof.selection()) == list(prof.probes[prof.index].selected)
    assert prof.pick.max_selected == max(counts)
    assert prof.seconds > 0
    np.testing.assert_array_equal(X, X_before)            # every probe hands X back

    # Tolerance 0 reads the largest selection, ties toward the smallest lambda_2.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        argmax = tsn.tikhonov_lambda2_power_rule(X, y, tFDR=0.2, tik_control=tik, trex_control=base,
                                                 seed=1, grid=[0.0, 1.0, 10.0], knee_tolerance=0.0)
    assert argmax.index == counts.index(max(counts))

    # The contract: a fixed-lambda_2 run at the pick with the same seed selects the same variables.
    tik_fixed = _tik_control(_group_mean_k(groups), prof.lambda2, prior_groups=groups)
    sel = tsn.TRexTikhonovSelector(X, y, tFDR=0.2, tik_control=tik_fixed, trex_control=base,
                                   seed=1, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel.select()
    assert sorted(sel.selected_indices) == list(prof.selection())

    with pytest.raises(ValueError, match="seed"):
        tsn.tikhonov_lambda2_power_rule(X, y, tik_control=tik, trex_control=base, seed=-1)
    with pytest.raises(ValueError):
        tsn.tikhonov_lambda2_power_rule(X, y, tik_control=tik, trex_control=base, knee_tolerance=-1.0)
    with pytest.raises(ValueError, match="tik_control"):
        tsn.tikhonov_lambda2_power_rule(X, y)
