"""Functional tests for the solver binding surface added in 0.2.0.

Covers: TIENETAug_Solver, the TENETAug constructor extensions
(scaling_mode / use_lars_inner) and its save/load, TENET.getAlpha,
the solver-level ScalingMode enum, setTolerance, and the
active-predictor/dummy index getters.
"""
import os

import numpy as np
import pytest

from trex_selector_neo.tsolvers import ScalingMode
from trex_selector_neo.tsolvers.lars_based import (
    TLARS_Solver,
    TENET_Solver,
    TENETAug_Solver,
    TIENETAug_Solver,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    n, p = 60, 10
    X = np.asfortranarray(rng.standard_normal((n, p)))
    # One group-aligned dummy layer (TIENETAug requires D.cols() % p == 0)
    D = np.asfortranarray(rng.standard_normal((n, p)))
    y = np.asfortranarray(
        X[:, 0] * 3.0 - X[:, 1] * 2.0 + rng.standard_normal(n) * 0.5
    )
    return X, D, y


def test_scaling_mode_enum_and_ctor_arg(data):
    X, D, y = data
    s_l2 = TLARS_Solver(X.copy(order="F"), D.copy(order="F"), y.copy(order="F"),
                        scaling_mode=ScalingMode.L2)
    s_z = TLARS_Solver(X.copy(order="F"), D.copy(order="F"), y.copy(order="F"),
                       scaling_mode=ScalingMode.ZSCORE)
    s_l2.executeStep(2)
    s_z.executeStep(2)
    # Different scaling conventions must both run; coefficient paths are
    # reported on the original scale and generally differ in later digits.
    assert s_l2.getNumSteps() > 0
    assert s_z.getNumSteps() > 0


def test_set_tolerance_and_active_index_getters(data):
    X, D, y = data
    s = TLARS_Solver(X, D, y)
    s.setTolerance(1e-12)
    s.executeStep(2)

    predictors = s.getActivePredictorIndices()
    dummies = s.getActiveDummyIndices()
    dummy_start = s.getDummyStartIndex()

    assert all(idx < dummy_start for idx in predictors)
    assert all(idx >= dummy_start for idx in dummies)
    assert len(predictors) + len(dummies) == s.getNumActives()


def test_tenet_get_alpha(data):
    X, D, y = data
    t = TENET_Solver(X, D, y, lambda2=0.5)
    # Before the path runs there is no lambda1 yet -> alpha defaults to 1.0
    assert t.getAlpha() == pytest.approx(1.0)
    t.executeStep(2)
    # With lambda2 > 0 and a fitted path, alpha = l1 / (l1 + l2) < 1
    alpha = t.getAlpha()
    assert 0.0 <= alpha < 1.0


def test_tenet_aug_ctor_extensions_and_save_load(tmp_path, data):
    X, D, y = data
    ta = TENETAug_Solver(X, D, y, lambda2=0.5,
                         scaling_mode=ScalingMode.L2, use_lars_inner=True)
    ta.executeStep(2)
    assert ta.getNumSteps() > 0

    filepath = os.path.join(tmp_path, "tenet_aug.bin")
    ta.save(filepath)
    ta2 = TENETAug_Solver(X, D, y, lambda2=0.5)
    ta2.load(filepath)
    assert np.allclose(ta.getBetaPath(), ta2.getBetaPath())


def test_tienet_aug_solver(tmp_path, data):
    X, D, y = data
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)

    ti = TIENETAug_Solver(X, D, y, lambda2=0.5, groups=groups)
    ti.executeStep(2)

    assert ti.solverTypeToString() == "TIENETAug"
    assert ti.getNumGroups() == 5
    assert ti.getLambda2() == pytest.approx(0.5)
    assert (ti.getGroups() == groups).all()
    assert ti.getNumSteps() > 0

    filepath = os.path.join(tmp_path, "tienet_aug.bin")
    ti.save(filepath)
    ti2 = TIENETAug_Solver(X, D, y, lambda2=0.5, groups=groups)
    ti2.load(filepath)
    assert np.allclose(ti.getBetaPath(), ti2.getBetaPath())


def test_tienet_aug_rejects_misaligned_dummies(data):
    X, _, y = data
    rng = np.random.default_rng(7)
    D_bad = np.asfortranarray(rng.standard_normal((X.shape[0], 5)))
    groups = np.zeros(X.shape[1], dtype=np.int32)
    with pytest.raises(ValueError):
        TIENETAug_Solver(X, D_bad, y, lambda2=0.5, groups=groups)


def test_eval_counts_and_f1():
    from trex_selector_neo.utils import (
        true_positives_count,
        false_positives_count,
        false_negatives_count,
        f1_score,
    )
    assert true_positives_count([0, 1, 5], [0, 1, 2]) == 2
    assert false_positives_count([0, 1, 5], [0, 1, 2]) == 1
    assert false_negatives_count([0, 1, 5], [0, 1, 2]) == 1
    assert f1_score(0.5, 0.5) == pytest.approx(0.5)
    assert f1_score(0.0, 0.0) == 0.0


def test_convert_to_memory_mapped(tmp_path, data):
    from trex_selector_neo.utils import convert_to_memory_mapped

    X, _, _ = data
    filepath = os.path.join(tmp_path, "x.bin")
    convert_to_memory_mapped(X, filepath)
    assert os.path.getsize(filepath) == X.shape[0] * X.shape[1] * 8
