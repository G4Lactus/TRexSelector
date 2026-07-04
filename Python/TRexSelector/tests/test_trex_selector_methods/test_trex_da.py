"""
Tests for TRexDASelector.
"""
import numpy as np
import pytest
import trex_selector


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector.TRexDASelector(X, y, trex_control=fast_control, verbose=False)
    assert sel is not None


def test_dimension_mismatch_raises(signal_data, fast_control):
    X, y, n, p = signal_data
    y_bad = np.random.randn(n + 10)
    with pytest.raises(Exception):
        trex_selector.TRexDASelector(X, y_bad, trex_control=fast_control, verbose=False)


# ---------------------------------------------------------------------------
# Execution and result fields
# ---------------------------------------------------------------------------

def test_execution_and_result(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector.TRexDASelector(X, y, trex_control=fast_control, verbose=False)
    res = sel.select()

    assert res is not None
    assert isinstance(sel.selected_var, np.ndarray)
    assert isinstance(sel.selected_indices, list)

    # 0-based indices in range
    if sel.selected_indices:
        assert all(0 <= i < p for i in sel.selected_indices)


def test_da_result_fields(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector.TRexDASelector(X, y, trex_control=fast_control, verbose=False)
    da_res = sel.select()
    assert da_res is not None
    assert isinstance(da_res.rho_thresh, float)
    assert isinstance(da_res.rho_grid, np.ndarray)
    assert isinstance(da_res.method, trex_selector.DAMethod)
    assert isinstance(da_res.cor_coef, float)

    # BT arrays may be empty (depends on method) but must be list-like
    assert isinstance(da_res.fdp_hat_array_bt, list)
    assert isinstance(da_res.phi_array_bt, list)


def test_da_control_parameter_fields():
    ctrl = trex_selector.TRexDAControlParameter()
    ctrl.rho_thr_DA = 0.5
    assert ctrl.rho_thr_DA == pytest.approx(0.5)

    # bt_selection_mode
    ctrl.bt_selection_mode = trex_selector.BTSelectionMode.RFaithful
    assert ctrl.bt_selection_mode == trex_selector.BTSelectionMode.RFaithful


# ---------------------------------------------------------------------------
# Properties inherited from TRexSelector base
# ---------------------------------------------------------------------------

def test_base_properties(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector.TRexDASelector(X, y, trex_control=fast_control, verbose=False)
    sel.select()

    assert isinstance(sel.tFDR, float)
    assert isinstance(sel.T_stop, int)
    assert isinstance(sel.L, int)
    assert isinstance(sel.fdp_hat_mat, np.ndarray)
    assert isinstance(sel.phi_mat, np.ndarray)


# ---------------------------------------------------------------------------
# DAMethod variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", [
    trex_selector.DAMethod.AR1,
    trex_selector.DAMethod.EQUI,
    trex_selector.DAMethod.BT,
])
def test_da_method_variants(signal_data, method):
    X, y, n, p = signal_data
    ctrl = trex_selector.TRexControlParameter()
    ctrl.K = 3
    da_ctrl = trex_selector.TRexDAControlParameter()
    da_ctrl.method = method
    sel = trex_selector.TRexDASelector(X, y, da_control=da_ctrl,
                                        trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None
    assert res.method == method


def test_da_method_bt_r_array(signal_data):
    """Bootstrap method populates r_array_bt."""
    X, y, n, p = signal_data
    ctrl = trex_selector.TRexControlParameter()
    ctrl.K = 3
    da_ctrl = trex_selector.TRexDAControlParameter()
    da_ctrl.method = trex_selector.DAMethod.BT
    sel = trex_selector.TRexDASelector(X, y, da_control=da_ctrl,
                                        trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None
    # r_array_bt should be a non-None array when BT method is used
    assert res.r_array_bt is not None


def test_da_prior_groups(signal_data):
    """PRIOR_GROUPS method accepts group assignment list."""
    X, y, n, p = signal_data
    ctrl = trex_selector.TRexControlParameter()
    ctrl.K = 3
    da_ctrl = trex_selector.TRexDAControlParameter()
    da_ctrl.method = trex_selector.DAMethod.PRIOR_GROUPS
    # p=20 predictors; assign them to 4 groups
    da_ctrl.prior_groups = [[i % 4 for i in range(p)]]  # one hierarchy level, 4 groups
    sel = trex_selector.TRexDASelector(X, y, da_control=da_ctrl,
                                        trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None
