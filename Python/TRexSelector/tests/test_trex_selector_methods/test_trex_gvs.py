"""
Tests for TRexGVSSelector.
"""
import numpy as np
import pytest
import trex_selector

# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_instantiation(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
    assert sel is not None


def test_dimension_mismatch_raises(signal_data, fast_control):
    X, y, n, p = signal_data
    y_bad = np.random.randn(n + 10)
    with pytest.raises(Exception):
        trex_selector.TRexGVSSelector(X, y_bad, trex_control=fast_control, verbose=False)


# ---------------------------------------------------------------------------
# Execution and result fields
# ---------------------------------------------------------------------------

def test_execution_and_result(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
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
    gvs_ctrl = trex_selector.TRexGVSControlParameter()

    # corr_max
    gvs_ctrl.corr_max = 0.7
    assert gvs_ctrl.corr_max == pytest.approx(0.7)

    # hc_linkage — just check assignment doesn't raise
    from trex_selector._core.ml_methods.clustering import LinkageMethod
    gvs_ctrl.hc_linkage = LinkageMethod.Complete
    assert gvs_ctrl.hc_linkage == LinkageMethod.Complete
    # lambda_2
    gvs_ctrl.lambda_2 = 0.01
    assert gvs_ctrl.lambda_2 == pytest.approx(0.01)


def test_gvs_prior_groups(signal_data, fast_control):
    X, y, n, p = signal_data

    gvs_ctrl = trex_selector.TRexGVSControlParameter()
    gvs_ctrl.prior_groups = [i % 4 for i in range(p)]  # 5 groups of 4

    sel = trex_selector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                        trex_control=fast_control, verbose=False)
    res = sel.select()
    assert res is not None


def test_gvs_result_fields(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
    gvs_res = sel.select()
    assert gvs_res is not None
    assert isinstance(gvs_res.gvs_type, trex_selector.GVSType)


# ---------------------------------------------------------------------------
# Properties inherited from TRexSelector base
# ---------------------------------------------------------------------------

def test_base_properties(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
    sel.select()

    assert isinstance(sel.tFDR, float)
    assert isinstance(sel.T_stop, int)
    assert isinstance(sel.L, int)
    assert isinstance(sel.fdp_hat_mat, np.ndarray)


# ---------------------------------------------------------------------------
# GVSType variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gvs_type", [
    trex_selector.GVSType.EN,
    trex_selector.GVSType.IEN,
])
def test_gvs_type_variants(signal_data, gvs_type):
    X, y, n, p = signal_data
    ctrl = trex_selector.TRexControlParameter()
    ctrl.K = 3
    gvs_ctrl = trex_selector.TRexGVSControlParameter()
    gvs_ctrl.gvs_type = gvs_type
    sel = trex_selector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                         trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None
    assert res.gvs_type == gvs_type


def test_gvs_lambda2_used(signal_data, fast_control):
    X, y, n, p = signal_data
    sel = trex_selector.TRexGVSSelector(X, y, trex_control=fast_control, verbose=False)
    res = sel.select()
    assert isinstance(res.lambda2_used, float)
    assert res.lambda2_used > 0.0


# ---------------------------------------------------------------------------
# LinkageMethod variants (non-Ward only — Ward requires Euclidean)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("linkage", [
    pytest.param("Average", id="Average"),
    pytest.param("Complete", id="Complete"),
    pytest.param("Single", id="Single"),
])
def test_gvs_linkage_methods(signal_data, linkage):
    from trex_selector._core.ml_methods.clustering import LinkageMethod
    X, y, n, p = signal_data
    ctrl = trex_selector.TRexControlParameter()
    ctrl.K = 3
    gvs_ctrl = trex_selector.TRexGVSControlParameter()
    gvs_ctrl.hc_linkage = getattr(LinkageMethod, linkage)
    sel = trex_selector.TRexGVSSelector(X, y, gvs_control=gvs_ctrl,
                                         trex_control=ctrl, verbose=False)
    res = sel.select()
    assert res is not None
