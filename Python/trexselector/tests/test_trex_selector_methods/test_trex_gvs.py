"""
Tests for TRexGVSSelector.
"""
import numpy as np
import pytest
import trexselector

# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trexselector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
    assert sel is not None


def test_dimension_mismatch_raises(signal_data, fast_control):
    X, y, n, p = signal_data
    y_bad = np.random.randn(n + 10)
    with pytest.raises(Exception):
        trexselector.TRexGVSSelector(X, y_bad, trex_control=fast_control, verbose=False)


# ---------------------------------------------------------------------------
# Execution and result fields
# ---------------------------------------------------------------------------

def test_execution_and_result(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trexselector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
    res = sel.select()

    assert res is not None
    assert isinstance(sel.selected_var, np.ndarray)
    assert isinstance(sel.selected_indices, list)

    if sel.selected_indices:
        assert all(0 <= i < p for i in sel.selected_indices)


# ---------------------------------------------------------------------------
# GVS control parameter fields
# ---------------------------------------------------------------------------

def test_gvs_control_new_fields():
    gvs_ctrl = trexselector.TRexGVSControlParameter()

    # corr_max
    gvs_ctrl.corr_max = 0.7
    assert gvs_ctrl.corr_max == pytest.approx(0.7)

    # hc_linkage — just check assignment doesn't raise
    from trexselector._core.ml_methods.clustering import LinkageMethod
    gvs_ctrl.hc_linkage = LinkageMethod.Complete
    assert gvs_ctrl.hc_linkage == LinkageMethod.Complete
    # lambda_2
    gvs_ctrl.lambda_2 = 0.01
    assert gvs_ctrl.lambda_2 == pytest.approx(0.01)


def test_gvs_control_en_solver_and_cv_fields():
    gvs_ctrl = trexselector.TRexGVSControlParameter()

    # en_solver
    gvs_ctrl.en_solver = trexselector.ENSolverType.TENET_AUG
    assert gvs_ctrl.en_solver == trexselector.ENSolverType.TENET_AUG

    # lambda2_method
    gvs_ctrl.lambda2_method = trexselector.LambdaSelectionMethod.CV_MIN_CCD
    assert gvs_ctrl.lambda2_method == trexselector.LambdaSelectionMethod.CV_MIN_CCD

    # CV knobs
    gvs_ctrl.cv_n_folds = 5
    gvs_ctrl.cv_n_lambda = 250
    gvs_ctrl.cv_seed = 7
    assert gvs_ctrl.cv_n_folds == 5
    assert gvs_ctrl.cv_n_lambda == 250
    assert gvs_ctrl.cv_seed == 7

    # tenet_aug_use_lars
    gvs_ctrl.tenet_aug_use_lars = True
    assert gvs_ctrl.tenet_aug_use_lars is True


def test_gvs_en_tenet_aug_runs(signal_data, fast_control):
    X, y, n, p = signal_data
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = trexselector.GVSType.EN
    gvs_ctrl.en_solver = trexselector.ENSolverType.TENET_AUG
    sel = trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                        trex_control=fast_control, verbose=False)
    res = sel.select()
    assert res is not None
    assert res.gvs_type == trexselector.GVSType.EN


def test_gvs_prior_groups(signal_data, fast_control):
    X, y, n, p = signal_data

    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.prior_groups = [i % 4 for i in range(p)]  # 5 groups of 4

    sel = trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                        trex_control=fast_control, verbose=False)
    res = sel.select()
    assert res is not None


def test_gvs_result_fields(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trexselector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
    gvs_res = sel.select()
    assert gvs_res is not None
    assert isinstance(gvs_res.gvs_type, trexselector.GVSType)


# ---------------------------------------------------------------------------
# Properties inherited from TRexSelector base
# ---------------------------------------------------------------------------

def test_base_properties(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trexselector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
    sel.select()

    assert isinstance(sel.tFDR, float)
    assert isinstance(sel.T_stop, int)
    assert isinstance(sel.L, int)
    assert isinstance(sel.fdp_hat_mat, np.ndarray)


# ---------------------------------------------------------------------------
# GVSType variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gvs_type", [
    trexselector.GVSType.EN,
    trexselector.GVSType.IEN,
])
def test_gvs_type_variants(signal_data, gvs_type):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = gvs_type
    sel = trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                         trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None
    assert res.gvs_type == gvs_type


def test_gvs_lambda2_used(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trexselector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
    res = sel.select()
    assert isinstance(res.lambda2_used, float)
    assert res.lambda2_used > 0.0


# ---------------------------------------------------------------------------
# Solver derivation warning
# ---------------------------------------------------------------------------

def test_gvs_warns_on_non_matching_solver(signal_data):
    """IEN: solver_type now selects the IEN solver, so a non-IEN-family value
    warns and falls back to TIENET; an IEN-family value is honored silently."""
    import warnings
    X, y, n, p = signal_data
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = trexselector.GVSType.IEN

    ctrl = trexselector.TRexControlParameter()
    ctrl.solver_type = trexselector.SolverTypeForTRex.TOMP
    with pytest.warns(UserWarning, match="not an IEN-family solver"):
        trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                          trex_control=ctrl, verbose=False)

    # Default (TLARS) and matching solver stay silent
    for solver in (trexselector.SolverTypeForTRex.TLARS,
                   trexselector.SolverTypeForTRex.TIENET_AUG):
        ctrl_ok = trexselector.TRexControlParameter()
        ctrl_ok.solver_type = solver
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                              trex_control=ctrl_ok, verbose=False)


def test_gvs_warns_on_lambda2_zero(signal_data):
    """lambda_2 == 0 is the degenerate no-ridge case, not automatic CV; the
    wrapper must warn so users don't silently get pure T-LASSO."""
    X, y, n, p = signal_data
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.lambda_2 = 0.0
    with pytest.warns(UserWarning, match="degenerate no-ridge"):
        trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl, verbose=False)


def test_gvs_no_lambda2_warning_for_auto_or_positive(signal_data):
    """lambda_2 < 0 (auto/CV) and lambda_2 > 0 (fixed) must not warn."""
    import warnings
    X, y, n, p = signal_data
    for lam in (-1.0, 0.05):
        gvs_ctrl = trexselector.TRexGVSControlParameter()
        gvs_ctrl.lambda_2 = lam
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl, verbose=False)


# ---------------------------------------------------------------------------
# LinkageMethod variants (non-Ward only — Ward requires Euclidean)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("linkage", [
    pytest.param("Average", id="Average"),
    pytest.param("Complete", id="Complete"),
    pytest.param("Single", id="Single"),
])
def test_gvs_linkage_methods(signal_data, linkage):
    from trexselector._core.ml_methods.clustering import LinkageMethod
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.hc_linkage = getattr(LinkageMethod, linkage)
    sel = trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                         trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


# ---------------------------------------------------------------------------
# CCD family + IEN-geometry lambda_2 tuners (bindings-parity additions)
# ---------------------------------------------------------------------------

def test_lambda2_method_ien_entries_exist():
    m = trexselector.LambdaSelectionMethod
    for name in ("CV_1SE_IEN_CCD", "CV_MIN_IEN_CCD",
                 "CV_1SE_TIK_SVD", "CV_MIN_TIK_SVD"):
        assert hasattr(m, name)


def test_lambda2_method_track_resolution_and_restore(signal_data):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = trexselector.GVSType.IEN
    gvs_ctrl.lambda_2 = 1.0
    assert gvs_ctrl.lambda2_method == trexselector.LambdaSelectionMethod.CV_1SE_CCD
    trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                      trex_control=ctrl, verbose=False)
    # The C++ constructor resolves the default to the IEN-geometry tuner on
    # its own copy; the caller's object stays as set.
    assert gvs_ctrl.lambda2_method == trexselector.LambdaSelectionMethod.CV_1SE_CCD


def test_lambda2_method_ien_tuner_rejected_on_en_track(signal_data):
    X, y, n, p = signal_data
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = trexselector.GVSType.EN
    gvs_ctrl.lambda2_method = trexselector.LambdaSelectionMethod.CV_1SE_IEN_CCD
    with pytest.raises(ValueError, match="requires gvs_type = IEN"):
        trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl, verbose=False)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_gvs_non_finite_cv_dfmax_ratio_raises(signal_data, bad):
    X, y, n, p = signal_data
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = trexselector.GVSType.IEN
    gvs_ctrl.cv_dfmax_ratio = bad
    with pytest.raises(ValueError, match="cv_dfmax_ratio must be finite"):
        trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl, verbose=False)


def test_gvs_en_tcdenet_runs(signal_data):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.en_solver = trexselector.ENSolverType.TCDENET
    gvs_ctrl.lambda_2 = 1.0
    sel = trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                            trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


def test_gvs_ien_solver_choice_tcdienet(signal_data):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3
    ctrl.solver_type = trexselector.SolverTypeForTRex.TCDIENET
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = trexselector.GVSType.IEN
    gvs_ctrl.lambda_2 = 1.0
    sel = trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                            trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None


def test_gvs_ien_tuner_grids_and_diagnostics(signal_data):
    X, y, n, p = signal_data
    ctrl = trexselector.TRexControlParameter()
    ctrl.K = 3

    # Fixed lambda_2: no tuner ran.
    fixed = trexselector.TRexGVSControlParameter()
    fixed.gvs_type = trexselector.GVSType.IEN
    fixed.lambda_2 = 1.0
    sel = trexselector.TRexGVSSelector(X, y, gvs_control=fixed,
                                            trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res.lambda2_cv.ran is False
    assert sel.getLambda2CVDiagnostics().ran is False
    assert res.lambda2_cv.dfmax_ratio == 0.0     # no IEN-geometry tuner ran

    # IEN-geometry CV with reduced grids: the tuner ran and is reported.
    gvs_ctrl = trexselector.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = trexselector.GVSType.IEN
    gvs_ctrl.lambda_2 = -1.0
    gvs_ctrl.cv_n_lambda1 = 20
    gvs_ctrl.cv_n_lambda2 = 10
    assert gvs_ctrl.cv_n_lambda1 == 20 and gvs_ctrl.cv_n_lambda2 == 10
    assert gvs_ctrl.cv_dfmax_ratio == pytest.approx(0.8)   # default
    gvs_ctrl.cv_dfmax_ratio = 0.9
    assert gvs_ctrl.cv_dfmax_ratio == pytest.approx(0.9)
    sel = trexselector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                            trex_control=ctrl, verbose=False)
    res = sel.select()
    diag = res.lambda2_cv
    assert diag.ran is True
    assert diag.seconds >= 0.0
    assert diag.n_profile_at_edge >= 0
    assert res.lambda2_used > 0

    # The support-cap ratio reached the tuner and is recorded as passed.
    assert diag.dfmax_ratio == pytest.approx(0.9)
    assert isinstance(diag.profile_at_support_cap, bool)
    assert diag.profile_at_edge or not diag.profile_at_support_cap

    # Boundary-pick guard fields: the chosen pick on its grid.
    assert diag.grid_size == 10
    assert 0 <= diag.grid_index < diag.grid_size
    assert isinstance(diag.at_grid_edge, bool) and isinstance(diag.profile_at_edge, bool)
    assert diag.boundary_pick == (diag.at_grid_edge or diag.profile_at_edge)
    assert diag.at_grid_edge == (diag.grid_index in (0, diag.grid_size - 1))
