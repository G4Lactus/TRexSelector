"""
Tests for TRexSPCASelector (trex_selector_neo.TRexSPCASelector).
"""
import numpy as np
import pytest
from trex_selector_neo import (
    TRexSPCASelector,
    TRexSPCAControlParameter,
    SPCAMode,
    GVSType,
)


# ---------------------------------------------------------------------------
# Shared fixture — small dataset to keep tests fast
# ---------------------------------------------------------------------------

@pytest.fixture
def spca_fixture():
    rng = np.random.default_rng(42)
    n, p = 30, 10
    X = rng.standard_normal((n, p)).astype(np.float64)
    return X, n, p


M    = 2
TFDR = 0.2


def make_ctrl(**kwargs):
    ctrl = TRexSPCAControlParameter()
    ctrl.gvs_ctrl.trex_ctrl.K = kwargs.get("K", 3)
    ctrl.gvs_ctrl.trex_ctrl.max_dummy_multiplier = kwargs.get("max_dummy_multiplier", 3)
    ctrl.mode = kwargs.get("mode", SPCAMode.ActiveSet)
    if "lambda2_ridge_loadings" in kwargs:
        ctrl.lambda2_ridge_loadings = kwargs["lambda2_ridge_loadings"]
    if "gvs_type" in kwargs:
        ctrl.gvs_ctrl.gvs_type = kwargs["gvs_type"]
    if "lambda_2" in kwargs:
        ctrl.gvs_ctrl.lambda_2 = kwargs["lambda_2"]
    return ctrl


# ---------------------------------------------------------------------------
# Group 1: TRexSPCAControlParameter defaults
# ---------------------------------------------------------------------------

def test_spca_control_default_mode():
    ctrl = TRexSPCAControlParameter()
    assert ctrl.mode == SPCAMode.ActiveSet


def test_spca_control_default_lambda2_ridge_loadings():
    ctrl = TRexSPCAControlParameter()
    assert np.isclose(ctrl.lambda2_ridge_loadings, 1e-6)


def test_spca_control_default_gvs_type():
    ctrl = TRexSPCAControlParameter()
    assert ctrl.gvs_ctrl.gvs_type == GVSType.EN


def test_spca_control_default_lambda_2():
    ctrl = TRexSPCAControlParameter()
    # -1.0 is the "not supplied" sentinel that triggers auto-selection of lambda_2.
    assert ctrl.gvs_ctrl.lambda_2 == -1.0


def test_spca_control_thresholded_mode():
    ctrl = TRexSPCAControlParameter()
    ctrl.mode = SPCAMode.Thresholded
    assert ctrl.mode == SPCAMode.Thresholded


# ---------------------------------------------------------------------------
# Group 2: Basic execution
# ---------------------------------------------------------------------------

def test_spca_select_runs_activeset_en(spca_fixture):
    X, _, _ = spca_fixture
    ctrl = make_ctrl()
    sel = TRexSPCASelector(X, ctrl, seed=1)
    sel.select(M, TFDR)  # must not raise


def test_spca_select_runs_thresholded_en(spca_fixture):
    X, _, _ = spca_fixture
    ctrl = make_ctrl(mode=SPCAMode.Thresholded)
    sel = TRexSPCASelector(X, ctrl, seed=1)
    sel.select(M, TFDR)  # must not raise


def test_spca_select_runs_activeset_ien(spca_fixture):
    X, _, _ = spca_fixture
    ctrl = make_ctrl(gvs_type=GVSType.IEN)
    sel = TRexSPCASelector(X, ctrl, seed=1)
    sel.select(M, TFDR)  # must not raise


def test_spca_select_runs_with_lambda_2_bypass(spca_fixture):
    X, _, _ = spca_fixture
    ctrl = make_ctrl(lambda_2=0.01)
    sel = TRexSPCASelector(X, ctrl, seed=1)
    sel.select(M, TFDR)  # must not raise


# ---------------------------------------------------------------------------
# Group 3: Output dimensions
# ---------------------------------------------------------------------------

def test_spca_Z_shape(spca_fixture):
    X, n, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(M, TFDR)
    assert sel.Z.shape == (n, M)


def test_spca_V_shape(spca_fixture):
    X, _, p = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(M, TFDR)
    assert sel.V.shape == (p, M)


def test_spca_active_sets_length(spca_fixture):
    X, _, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(M, TFDR)
    assert len(sel.active_sets) == M


def test_spca_explained_variance_length(spca_fixture):
    X, _, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(M, TFDR)
    assert sel.adjusted_ev.shape == (M,)
    assert sel.cumulative_ev.shape == (M,)


def test_spca_M1_shapes(spca_fixture):
    X, n, p = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(1, TFDR)
    assert sel.Z.shape == (n, 1)
    assert sel.V.shape == (p, 1)
    assert len(sel.active_sets) == 1


# ---------------------------------------------------------------------------
# Group 4: Active set bounds (0-based indices)
# ---------------------------------------------------------------------------

def test_spca_active_set_indices_in_range(spca_fixture):
    X, _, p = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(M, TFDR)
    for as_m in sel.active_sets:
        for idx in as_m:
            assert 0 <= idx < p


# ---------------------------------------------------------------------------
# Group 5: Explained variance properties
# ---------------------------------------------------------------------------

def test_spca_adjusted_ev_non_negative(spca_fixture):
    X, _, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(M, TFDR)
    assert np.all(sel.adjusted_ev >= 0)


def test_spca_cumulative_ev_non_decreasing(spca_fixture):
    X, _, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(M, TFDR)
    assert np.all(np.diff(sel.cumulative_ev) >= -1e-10)


def test_spca_cumulative_ev_in_unit_interval(spca_fixture):
    X, _, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(M, TFDR)
    assert np.all(sel.cumulative_ev >= 0)
    assert np.all(sel.cumulative_ev <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Group 6: Data integrity — X restored after select()
# ---------------------------------------------------------------------------

def test_spca_X_restored_after_select(spca_fixture):
    X, _, _ = spca_fixture
    X_orig = X.copy()  # genuine deep copy
    sel = TRexSPCASelector(X, make_ctrl(), seed=1)
    sel.select(M, TFDR)
    assert np.allclose(X, X_orig, atol=1e-12)


# ---------------------------------------------------------------------------
# Group 7: Input validation
# ---------------------------------------------------------------------------

def test_spca_rejects_M_zero(spca_fixture):
    X, _, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl())
    with pytest.raises((ValueError, Exception)):
        sel.select(0, TFDR)


def test_spca_rejects_tFDR_zero(spca_fixture):
    X, _, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl())
    with pytest.raises((ValueError, Exception)):
        sel.select(M, 0.0)


def test_spca_rejects_tFDR_one(spca_fixture):
    X, _, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl())
    with pytest.raises((ValueError, Exception)):
        sel.select(M, 1.0)


def test_spca_properties_raise_before_select(spca_fixture):
    X, _, _ = spca_fixture
    sel = TRexSPCASelector(X, make_ctrl())
    with pytest.raises(RuntimeError):
        _ = sel.Z
