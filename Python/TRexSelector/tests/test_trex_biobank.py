"""
Tests for TRexBiobankScreeningSelector.
"""
import numpy as np
import pytest
import trex_selector
from trex_selector import (
    TRexBiobankScreeningSelector,
    BiobankScreenTRexControl,
    BiobankScreenTRexResult,
)


# ---------------------------------------------------------------------------
# BiobankScreenTRexControl defaults
# ---------------------------------------------------------------------------

def test_biobank_control_defaults():
    ctrl = BiobankScreenTRexControl()
    assert isinstance(ctrl.target_FDR_trex, float)
    assert isinstance(ctrl.lower_bound_FDR, float)
    assert isinstance(ctrl.upper_bound_FDR, float)
    assert ctrl.lower_bound_FDR <= ctrl.upper_bound_FDR


def test_biobank_control_assignment():
    ctrl = BiobankScreenTRexControl()
    ctrl.target_FDR_trex = 0.05
    ctrl.lower_bound_FDR = 0.01
    ctrl.upper_bound_FDR = 0.2
    assert ctrl.target_FDR_trex == pytest.approx(0.05)
    assert ctrl.lower_bound_FDR == pytest.approx(0.01)
    assert ctrl.upper_bound_FDR == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Single phenotype (1-D y) — screenPhenotype()
# ---------------------------------------------------------------------------

def test_instantiation_1d(signal_data):
    X, y, n, p = signal_data
    sel = TRexBiobankScreeningSelector(X, y, verbose=False)
    assert sel is not None


def test_select_returns_result(signal_data):
    X, y, n, p = signal_data
    sel = TRexBiobankScreeningSelector(X, y, verbose=False)
    res = sel.select()
    assert res is not None
    assert isinstance(res, BiobankScreenTRexResult)


def test_result_phenotype_index(signal_data):
    X, y, n, p = signal_data
    sel = TRexBiobankScreeningSelector(X, y, verbose=False)
    res = sel.select()
    assert res.phenotype_index == 0


def test_result_selected_indices(signal_data):
    X, y, n, p = signal_data
    sel = TRexBiobankScreeningSelector(X, y, verbose=False)
    res = sel.select()
    assert isinstance(res.selected_indices, list)
    if res.selected_indices:
        assert all(0 <= i < p for i in res.selected_indices)


def test_result_estimated_fdr(signal_data):
    X, y, n, p = signal_data
    sel = TRexBiobankScreeningSelector(X, y, verbose=False)
    res = sel.select()
    assert isinstance(res.estimated_FDR, float)
    assert 0.0 <= res.estimated_FDR <= 1.0


def test_result_method_used(signal_data):
    X, y, n, p = signal_data
    sel = TRexBiobankScreeningSelector(X, y, verbose=False)
    res = sel.select()
    assert isinstance(res.method_used, str)
    assert len(res.method_used) > 0


def test_result_used_fallback_trex(signal_data):
    X, y, n, p = signal_data
    sel = TRexBiobankScreeningSelector(X, y, verbose=False)
    res = sel.select()
    assert isinstance(res.used_fallback_trex, bool)


def test_result_auxiliary_index_lists(signal_data):
    X, y, n, p = signal_data
    sel = TRexBiobankScreeningSelector(X, y, verbose=False)
    res = sel.select()
    assert isinstance(res.selected_indices_screen_ordinary, list)
    assert isinstance(res.selected_indices_screen_bootstrap, list)


# ---------------------------------------------------------------------------
# Multiple phenotypes (2-D Y) — screenPhenotypes()
# ---------------------------------------------------------------------------

def test_instantiation_2d(signal_data):
    X, y, n, p = signal_data
    Y = np.column_stack([y, y])
    sel = TRexBiobankScreeningSelector(X, Y, verbose=False)
    assert sel is not None


def test_select_2d_returns_list(signal_data):
    X, y, n, p = signal_data
    Y = np.column_stack([y, y])
    sel = TRexBiobankScreeningSelector(X, Y, verbose=False)
    results = sel.select()
    assert isinstance(results, list)
    assert len(results) == 2


def test_select_2d_phenotype_indices(signal_data):
    X, y, n, p = signal_data
    Y = np.column_stack([y, y])
    sel = TRexBiobankScreeningSelector(X, Y, verbose=False)
    results = sel.select()
    indices = [r.phenotype_index for r in results]
    assert sorted(indices) == [0, 1]


def test_select_2d_result_types(signal_data):
    X, y, n, p = signal_data
    Y = np.column_stack([y, y])
    sel = TRexBiobankScreeningSelector(X, Y, verbose=False)
    results = sel.select()
    for res in results:
        assert isinstance(res, BiobankScreenTRexResult)
        assert isinstance(res.selected_indices, list)
        assert isinstance(res.estimated_FDR, float)
        assert 0.0 <= res.estimated_FDR <= 1.0
