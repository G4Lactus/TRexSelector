"""Functional tests for the solver binding surface added in 0.2.0.

Covers: TIENETAug_Solver, the TENETAug constructor extensions
(scaling_mode / use_lars_inner) and its save/load, TENET.getAlpha,
the solver-level ScalingMode enum, setTolerance, and the
active-predictor/dummy index getters.
"""
import os

import numpy as np
import pytest

from trexselector.tsolvers.ad import ScalingMode, SolverOptions
from trexselector.tsolvers.ad.lars_based import (
    TLARS_Solver,
    TENET_Solver,
    TENETAug_Solver,
    TIENETAug_Solver,
)
from trexselector.tsolvers.ad.omp_based import TOMP_Solver


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
                        SolverOptions(scaling_mode=ScalingMode.L2))
    s_z = TLARS_Solver(X.copy(order="F"), D.copy(order="F"), y.copy(order="F"),
                       SolverOptions(scaling_mode=ScalingMode.ZSCORE))
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
    t = TENET_Solver(X, D, y, SolverOptions(lambda2=0.5))
    # Before the path runs there is no lambda1 yet -> alpha defaults to 1.0
    assert t.getAlpha() == pytest.approx(1.0)
    t.executeStep(2)
    # With lambda2 > 0 and a fitted path, alpha = l1 / (l1 + l2) < 1
    alpha = t.getAlpha()
    assert 0.0 <= alpha < 1.0


def test_tenet_aug_ctor_extensions_and_save_load(tmp_path, data):
    X, D, y = data
    ta = TENETAug_Solver(X, D, y, SolverOptions(lambda2=0.5, scaling_mode=ScalingMode.L2,
                                                tenet_aug_use_lars=True))
    ta.executeStep(2)
    assert ta.getNumSteps() > 0

    filepath = os.path.join(tmp_path, "tenet_aug.bin")
    ta.save(filepath)
    ta2 = TENETAug_Solver(X, D, y, SolverOptions(lambda2=0.5))
    ta2.load(filepath)
    assert np.allclose(ta.getBetaPath(), ta2.getBetaPath())


def test_tienet_aug_solver(tmp_path, data):
    X, D, y = data
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)

    ti = TIENETAug_Solver(X, D, y, groups, SolverOptions(lambda2=0.5))
    ti.executeStep(2)

    assert ti.solverTypeToString() == "TIENETAug"
    assert ti.getNumGroups() == 5
    assert ti.getLambda2() == pytest.approx(0.5)
    assert (ti.getGroups() == groups).all()
    assert ti.getNumSteps() > 0

    filepath = os.path.join(tmp_path, "tienet_aug.bin")
    ti.save(filepath)
    ti2 = TIENETAug_Solver(X, D, y, groups, SolverOptions(lambda2=0.5))
    ti2.load(filepath)
    assert np.allclose(ti.getBetaPath(), ti2.getBetaPath())


def test_tienet_aug_rejects_misaligned_dummies(data):
    X, _, y = data
    rng = np.random.default_rng(7)
    D_bad = np.asfortranarray(rng.standard_normal((X.shape[0], 5)))
    groups = np.zeros(X.shape[1], dtype=np.int32)
    with pytest.raises(ValueError):
        TIENETAug_Solver(X, D_bad, y, groups, SolverOptions(lambda2=0.5))


def test_eval_counts_and_f1():
    from trexselector.utils import (
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
    from trexselector.utils import convert_to_memory_mapped

    X, _, _ = data
    filepath = os.path.join(tmp_path, "x.bin")
    convert_to_memory_mapped(X, filepath)
    assert os.path.getsize(filepath) == X.shape[0] * X.shape[1] * 8


# ---------------------------------------------------------------------------
# setExchangeableTie (greedy solvers)
# ---------------------------------------------------------------------------

def _clustered_data(seed=3):
    """Design with a tight correlated cluster so the tie set is non-trivial."""
    rng = np.random.default_rng(seed)
    n, p = 80, 8
    base = rng.standard_normal(n)
    X = np.empty((n, p))
    for j in range(4):  # cluster: rho ~ 0.95
        X[:, j] = base + 0.3 * rng.standard_normal(n)
    X[:, 4:] = rng.standard_normal((n, p - 4))
    D = rng.standard_normal((n, p))
    y = base * 3.0 + rng.standard_normal(n) * 0.5
    return (np.asfortranarray(X), np.asfortranarray(D),
            np.asfortranarray(y))


@pytest.mark.parametrize("make_solver", [
    lambda X, D, y: __import__("trexselector").tsolvers.ad.omp_based.TOMP_Solver(X, D, y),
    lambda X, D, y: __import__("trexselector").tsolvers.ad.afs_based.TAFS_Solver(X, D, y),
])
def test_set_exchangeable_tie_reproducible(make_solver):
    X, D, y = _clustered_data()

    def run(alpha, seed):
        s = make_solver(X.copy(order="F"), D.copy(order="F"), y.copy(order="F"))
        s.setExchangeableTie(alpha, 0.5)
        s.setTieSeed(seed)
        s.executeStep(0, True)
        return [a for step in s.getActions() for a in step]

    # Same seed + same alpha => identical path (RNG-driven but reproducible).
    assert run(0.25, 123) == run(0.25, 123)
    # alpha <= 0 is a no-op: matches a run that never enables the feature.
    s_ref = make_solver(X.copy(order="F"), D.copy(order="F"), y.copy(order="F"))
    s_ref.executeStep(0, True)
    ref = [a for step in s_ref.getActions() for a in step]
    assert run(0.0, 123) == ref


def test_solver_options_keywords_and_fields(data):
    """SolverOptions: keyword constructor, unknown keywords rejected, fields round-trip,
    and the options reach the solver."""
    X, D, y = data
    o = SolverOptions(lambda2=0.25, normalize=False, tol=1e-9, tie_seed=7)
    assert o.lambda2 == 0.25 and o.normalize is False and o.intercept is True
    assert o.tol == 1e-9 and o.tie_seed == 7
    o.exch_tie_alpha = 0.25
    assert o.exch_tie_alpha == 0.25
    with pytest.raises(TypeError):
        SolverOptions(lambda_2=0.25)
    t = TENET_Solver(X, D, y, o)
    t.executeStep(2)
    assert t.getNumSteps() > 0
    assert SolverOptions().precentered is False
    assert SolverOptions().tie_seed == 0  # fixed default: reproducible tie-breaking
    assert SolverOptions(precentered=True).precentered is True


def _exactly_centered_integers(rng, n, p):
    """Integer columns with exactly zero sums: a solver's own centering is a bitwise no-op."""
    M = rng.integers(-6, 7, size=(n - 1, p)).astype(float)
    return np.asfortranarray(np.vstack([M, -M.sum(axis=0)]))


def test_precentered_counts_intercept_like_intercept():
    """precentered: a reader on caller-centered data counts the intercept like
    intercept=True (same path, DoF and Cp) and caps the active set at n - 1."""
    rng = np.random.default_rng(7)
    n = 20
    X = _exactly_centered_integers(rng, n, 30)
    D = _exactly_centered_integers(rng, n, 30)
    y = np.asfortranarray(X[:, 1] * 3.0 - X[:, 4] * 2.0 + _exactly_centered_integers(rng, n, 1)[:, 0])

    fitted = TOMP_Solver(X.copy(order="F"), D.copy(order="F"), y.copy(),
                         SolverOptions(normalize=False, intercept=True, tie_seed=7))
    reader = TOMP_Solver(X, D, y,
                         SolverOptions(normalize=False, intercept=False, precentered=True, tie_seed=7))
    fitted.executeStep(30, True)
    reader.executeStep(30, True)

    assert reader.getActions() == fitted.getActions()
    assert list(reader.getDoF()) == list(fitted.getDoF())
    np.testing.assert_array_equal(np.asarray(reader.getCp()), np.asarray(fitted.getCp()))
    assert reader.getDoF()[0] == 1
    assert reader.getNumActives() <= n - 1


def test_n_threads_option_and_setters():
    """n_threads is the solver's own thread budget (default 1, at least 1); the path
    is bitwise independent of it, and values below 1 are rejected."""
    from trexselector.tsolvers.ad.lars_based import TLARS_Solver

    rng = np.random.default_rng(5)
    n, p, num_dummies = 400, 1500, 1500          # 1.2 million entries per pass
    X = np.asfortranarray(rng.standard_normal((n, p)))
    D = np.asfortranarray(rng.standard_normal((n, num_dummies)))
    y = X[:, :5] @ np.array([3.0, -2.0, 2.0, -3.0, 2.5]) + rng.standard_normal(n)
    assert SolverOptions().n_threads == 1

    def run(n_threads):
        s = TLARS_Solver(X.copy(order="F"), D.copy(order="F"), y.copy(),
                         SolverOptions(n_threads=n_threads))
        s.executeStep(4, True)
        return s.getNumThreads(), s.getActions(), s.getBetaPath()

    t1, a1, b1 = run(1)
    t4, a4, b4 = run(4)
    assert (t1, t4) == (1, 4)
    assert a4 == a1
    assert np.array_equal(b4, b1)

    with pytest.raises(ValueError):
        TLARS_Solver(X.copy(order="F"), D.copy(order="F"), y.copy(), SolverOptions(n_threads=0))
    s = TLARS_Solver(X.copy(order="F"), D.copy(order="F"), y.copy(), SolverOptions(n_threads=3))
    assert s.getNumThreads() == 3
    s.setNumThreads(2)
    assert s.getNumThreads() == 2
    with pytest.raises(ValueError):
        s.setNumThreads(-1)
    with pytest.raises(ValueError):
        s.setNumThreads(0)
