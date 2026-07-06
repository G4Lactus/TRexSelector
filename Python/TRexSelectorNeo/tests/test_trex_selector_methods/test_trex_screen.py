"""
Tests for TRexScreeningSelector.
"""
import numpy as np
import pytest
import trex_selector_neo


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector_neo.TRexScreeningSelector(X, y, trex_control=fast_control, verbose=False)
    assert sel is not None


def test_dimension_mismatch_raises(signal_data, fast_control):
    X, y, n, p = signal_data
    y_bad = np.random.randn(n + 10)
    with pytest.raises(Exception):
        trex_selector_neo.TRexScreeningSelector(X, y_bad, trex_control=fast_control, verbose=False)


# ---------------------------------------------------------------------------
# Execution and result fields
# ---------------------------------------------------------------------------

def test_execution_and_result(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector_neo.TRexScreeningSelector(X, y, trex_control=fast_control, verbose=False)
    res = sel.select()

    assert res is not None
    assert isinstance(sel.selected_var, np.ndarray)
    assert isinstance(sel.selected_indices, list)

    if sel.selected_indices:
        assert all(0 <= i < p for i in sel.selected_indices)


def test_screen_result_fields(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector_neo.TRexScreeningSelector(X, y, trex_control=fast_control, verbose=False)
    screen_res = sel.select()
    assert screen_res is not None
    assert isinstance(screen_res.estimated_FDR, float)
    assert isinstance(screen_res.used_bootstrap, bool)
    assert isinstance(screen_res.confidence_level, float)
    assert isinstance(screen_res.beta_mat, np.ndarray)
    assert isinstance(screen_res.dummy_betas, np.ndarray)
    assert isinstance(screen_res.estimated_correlation, float)


# ---------------------------------------------------------------------------
# Screen control parameter fields
# ---------------------------------------------------------------------------

def test_screen_control_n_blocks():
    ctrl = trex_selector_neo.ScreenTRexControlParameter()
    ctrl.n_blocks = 5
    assert ctrl.n_blocks == 5


def test_screen_control_fields():
    ctrl = trex_selector_neo.ScreenTRexControlParameter()
    ctrl.use_bootstrap_CI = False
    ctrl.R_boot = 50
    ctrl.cor_coef = 0.5
    assert ctrl.use_bootstrap_CI is False
    assert ctrl.R_boot == 50
    assert ctrl.cor_coef == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Properties inherited from TRexSelector base
# ---------------------------------------------------------------------------

def test_base_properties(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector_neo.TRexScreeningSelector(X, y, trex_control=fast_control, verbose=False)
    sel.select()

    assert isinstance(sel.tFDR, float)
    assert isinstance(sel.T_stop, int)
    assert isinstance(sel.fdp_hat_mat, np.ndarray)


# ---------------------------------------------------------------------------
# ScreenTRexMethod variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", [
    trex_selector_neo.ScreenTRexMethod.TREX,
    trex_selector_neo.ScreenTRexMethod.TREX_DA_AR1,
    trex_selector_neo.ScreenTRexMethod.TREX_DA_EQUI,
    trex_selector_neo.ScreenTRexMethod.TREX_DA_BLOCK_EQUI,
])
def test_screen_method_variants(signal_data, method):
    X, y, n, p = signal_data
    ctrl = trex_selector_neo.TRexControlParameter()
    ctrl.K = 3
    screen_ctrl = trex_selector_neo.ScreenTRexControlParameter()
    screen_ctrl.trex_method = method
    if method == trex_selector_neo.ScreenTRexMethod.TREX_DA_BLOCK_EQUI:
        screen_ctrl.n_blocks = 2
    sel = trex_selector_neo.TRexScreeningSelector(X, y, screen_control=screen_ctrl,
                                               trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None
    assert isinstance(res.estimated_FDR, float)
    assert 0.0 <= res.estimated_FDR <= 1.0


def test_screen_auto_estimate_cor(signal_data):
    """cor_coef = -2.0 triggers auto-estimation of the correlation coefficient."""
    X, y, n, p = signal_data
    ctrl = trex_selector_neo.TRexControlParameter()
    ctrl.K = 3
    screen_ctrl = trex_selector_neo.ScreenTRexControlParameter()
    screen_ctrl.trex_method = trex_selector_neo.ScreenTRexMethod.TREX_DA_AR1
    screen_ctrl.cor_coef = -2.0  # sentinel: auto-estimate
    sel = trex_selector_neo.TRexScreeningSelector(X, y, screen_control=screen_ctrl,
                                               trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None
    # estimated_correlation should have been filled in
    assert isinstance(res.estimated_correlation, float)
