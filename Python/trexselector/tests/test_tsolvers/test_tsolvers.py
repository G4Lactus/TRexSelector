"""
Tests for all T-Solver classes: common API + solver-specific getters.
"""
import numpy as np
import pytest
from trexselector.tsolvers.ad.lars_based import (
    TLARS_Solver,
    TLASSO_Solver,
    TSTEPWISE_Solver,
    TSTAGEWISE_Solver,
    TENET_Solver,
    TENETAug_Solver,
)
from trexselector.tsolvers.ad.omp_based import (
    TOMP_Solver,
    TGP_Solver,
    TACGP_Solver,
    TMP_Solver,
    TOOLS_Solver,
    TNCGMP_Solver,
    NCGMPVariant,
)
from trexselector.tsolvers.ad.afs_based import TAFS_Solver
from trexselector.tsolvers.ad import SolverOptions
from trexselector.tsolvers.ad.lars_based import TIENET_Solver, TIKNET_Solver
from trexselector.tsolvers.ad.cd_based import (
    TCDLASSO_Solver, TCDENET_Solver, TCDIENET_Solver, TCDTIKNET_Solver)


def _groups_for(p):
    # 0-based contiguous ids: pairs, last variable alone when p is odd
    return np.array([j // 2 for j in range(p)], dtype=np.int32)


def _group_mean_k(p):
    g = _groups_for(p)
    K = np.zeros((p, p))
    for m in np.unique(g):
        idx = np.flatnonzero(g == m)
        K[np.ix_(idx, idx)] = 1.0 / idx.size
    return K


# ---------------------------------------------------------------------------
# Helpers: build a solver instance from solver_data fixture
# ---------------------------------------------------------------------------

def make_tlars(X, D, y):
    return TLARS_Solver(X, D, y)

def make_tlasso(X, D, y):
    return TLASSO_Solver(X, D, y)

def make_tstepwise(X, D, y):
    return TSTEPWISE_Solver(X, D, y)

def make_tstagewise(X, D, y):
    return TSTAGEWISE_Solver(X, D, y)

def make_tenet(X, D, y):
    return TENET_Solver(X, D, y, SolverOptions(lambda2=0.1))

def make_tenet_aug(X, D, y):
    return TENETAug_Solver(X, D, y, SolverOptions(lambda2=0.1))

def make_tomp(X, D, y):
    return TOMP_Solver(X, D, y)

def make_tgp(X, D, y):
    return TGP_Solver(X, D, y)

def make_tacgp(X, D, y):
    return TACGP_Solver(X, D, y)

def make_tmp(X, D, y):
    return TMP_Solver(X, D, y)

def make_tools(X, D, y):
    return TOOLS_Solver(X, D, y)

def make_tncgmp(X, D, y):
    return TNCGMP_Solver(X, D, y, SolverOptions(ncgmp_variant=NCGMPVariant.LineSearch))

def make_tafs(X, D, y):
    return TAFS_Solver(X, D, y, SolverOptions(rho_afs=1.0))

# Native IEN, the sparse-K solvers, the CD family.
def make_tienet(X, D, y):
    return TIENET_Solver(X, D, y, _groups_for(X.shape[1]), SolverOptions(lambda2=0.5))

def make_tiknet(X, D, y):
    return TIKNET_Solver(X, D, y, _group_mean_k(X.shape[1]), SolverOptions(lambda2=0.5))

def make_tcdlasso(X, D, y):
    return TCDLASSO_Solver(X, D, y)

def make_tcdenet(X, D, y):
    return TCDENET_Solver(X, D, y, SolverOptions(lambda2=0.5))

def make_tcdienet(X, D, y):
    return TCDIENET_Solver(X, D, y, _groups_for(X.shape[1]), SolverOptions(lambda2=0.5))

def make_tcdtiknet(X, D, y):
    return TCDTIKNET_Solver(X, D, y, _group_mean_k(X.shape[1]),
                            SolverOptions(lambda2=0.5, fold_dummies=True))


STANDARD_SOLVER_FACTORIES = [
    make_tlars,
    make_tlasso,
    make_tstepwise,
    make_tstagewise,
    make_tenet,
    make_tomp,
    make_tgp,
    make_tacgp,
    make_tmp,
    make_tools,
    make_tncgmp,
    make_tafs,
    make_tienet,
    make_tiknet,
    make_tcdlasso,
    make_tcdenet,
    make_tcdienet,
    make_tcdtiknet,
]


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_instantiation(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    assert solver is not None


def test_tenet_requires_lambda2(solver_data):
    """TENET needs a lambda2 positional argument."""
    X, D, y, n, p = solver_data
    # lambda2 lives in SolverOptions (default 0: the lasso path); a negative
    # value is rejected by the solver.
    assert TENET_Solver(X, D, y) is not None
    with pytest.raises(ValueError):
        TENET_Solver(X, D, y, SolverOptions(lambda2=-0.1))


def test_tncgmp_line_search(solver_data):
    X, D, y, n, p = solver_data
    solver = TNCGMP_Solver(X, D, y, SolverOptions(ncgmp_variant=NCGMPVariant.LineSearch))
    assert solver is not None


def test_tncgmp_fully_corrective(solver_data):
    X, D, y, n, p = solver_data
    solver = TNCGMP_Solver(X, D, y, SolverOptions(ncgmp_variant=NCGMPVariant.FullyCorrective))
    assert solver is not None


def test_tenet_aug_requires_lambda2(solver_data):
    """TENETAug needs a lambda2 positional argument."""
    X, D, y, n, p = solver_data
    assert TENETAug_Solver(X, D, y) is not None
    with pytest.raises(ValueError):
        TENETAug_Solver(X, D, y, SolverOptions(lambda2=-0.1))


def test_tenet_aug_execute_and_beta(solver_data):
    X, D, y, n, p = solver_data
    solver = make_tenet_aug(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    beta = solver.getBeta()
    assert isinstance(beta, np.ndarray)
    assert isinstance(solver.solverTypeToString(), str)


# ---------------------------------------------------------------------------
# Common API after executeStep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_execute_step_runs(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_get_num_steps(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(early_stop=False)
    assert solver.getNumSteps() >= 1


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_solver_type_to_string(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    s = solver.solverTypeToString()
    assert isinstance(s, str)
    assert len(s) > 0


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_get_beta_shape(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    beta = solver.getBeta()
    assert isinstance(beta, np.ndarray)
    # Beta covers all p real + p dummy predictors
    assert beta.shape == (p + p,)


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_get_beta_path_shape(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(early_stop=False)
    path = solver.getBetaPath()
    assert isinstance(path, np.ndarray)
    assert path.ndim == 2
    assert path.shape[0] == p + p  # rows = total predictors
    assert path.shape[1] == solver.getNumSteps() + 1  # +1 for initial zero state


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_get_rss(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    rss = solver.getRSS()
    assert len(rss) >= 1
    assert all(v >= 0 for v in rss)


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_get_dummy_start_index(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    idx = solver.getDummyStartIndex()
    # Dummy columns start after the p real predictors
    assert idx == p


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_get_actives_and_inactives(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    actives = solver.getActives()
    inactives = solver.getInactives()
    total = p + p
    assert set(actives) | set(inactives) == set(range(total))


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_is_connected(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    result = solver.isConnected()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Solver-specific getters (LARS-based)
# ---------------------------------------------------------------------------

def test_tlars_get_lambda(solver_data):
    X, D, y, n, p = solver_data
    solver = TLARS_Solver(X, D, y)
    solver.executeStep(early_stop=False)
    lam = solver.getLambda()
    assert isinstance(lam, list)
    assert len(lam) >= 1
    assert all(v >= 0 for v in lam)


def test_tlars_get_kahan_refresh_interval(solver_data):
    X, D, y, n, p = solver_data
    solver = TLARS_Solver(X, D, y)
    interval = solver.getKahanRefreshInterval()
    assert isinstance(interval, int)
    assert interval >= 1


def test_tlars_set_kahan_refresh_interval(solver_data):
    X, D, y, n, p = solver_data
    solver = TLARS_Solver(X, D, y)
    solver.setKahanRefreshInterval(20)
    assert solver.getKahanRefreshInterval() == 20


def test_tlasso_get_num_removals(solver_data):
    X, D, y, n, p = solver_data
    solver = TLASSO_Solver(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    assert isinstance(solver.getNumRemovals(), int)
    assert solver.getNumRemovals() >= 0


def test_tlasso_get_cycling_ratio(solver_data):
    X, D, y, n, p = solver_data
    solver = TLASSO_Solver(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    ratio = solver.getCyclingRatio()
    assert isinstance(ratio, float)
    assert ratio >= 0.0


def test_tstagewise_get_num_removals(solver_data):
    X, D, y, n, p = solver_data
    solver = TSTAGEWISE_Solver(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    assert isinstance(solver.getNumRemovals(), int)
    assert solver.getNumRemovals() >= 0


def test_tstagewise_get_cycling_ratio(solver_data):
    X, D, y, n, p = solver_data
    solver = TSTAGEWISE_Solver(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    ratio = solver.getCyclingRatio()
    assert isinstance(ratio, float)
    assert ratio >= 0.0


def test_tenet_get_num_removals(solver_data):
    X, D, y, n, p = solver_data
    solver = TENET_Solver(X, D, y, SolverOptions(lambda2=0.1))
    solver.executeStep(T_stop=0, early_stop=True)
    assert isinstance(solver.getNumRemovals(), int)


def test_tenet_get_cycling_ratio(solver_data):
    X, D, y, n, p = solver_data
    solver = TENET_Solver(X, D, y, SolverOptions(lambda2=0.1))
    solver.executeStep(T_stop=0, early_stop=True)
    assert isinstance(solver.getCyclingRatio(), float)


# ---------------------------------------------------------------------------
# getBeta with explicit step index
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [make_tlars, make_tlasso, make_tenet],
                         ids=["TLARS", "TLASSO", "TENET"])
def test_get_beta_at_step_0(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    beta_0 = solver.getBeta(step=0)
    # At step 0 (null model) beta should be all zeros
    assert isinstance(beta_0, np.ndarray)
    assert beta_0.shape == (p + p,)


@pytest.mark.parametrize("factory", [make_tlars, make_tlasso, make_tenet],
                         ids=["TLARS", "TLASSO", "TENET"])
def test_get_intercept(solver_data, factory):
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(T_stop=0, early_stop=True)
    intercept = solver.getIntercept()
    assert isinstance(intercept, float)


# ---------------------------------------------------------------------------
# Step-indexed getBeta / getIntercept and betaPath invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_get_beta_step_zero_is_zero_vector(solver_data, factory):
    """getBeta(step=0) must return the all-zero null model for every solver."""
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(early_stop=False)
    beta_0 = solver.getBeta(step=0)
    assert isinstance(beta_0, np.ndarray)
    assert beta_0.shape == (p + p,)
    assert np.allclose(beta_0, 0.0), f"{factory.__name__}: getBeta(0) is not zero: {beta_0}"


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_get_intercept_step_zero_is_zero(solver_data, factory):
    """getIntercept(step=0) is the null-model intercept (y_mean after centering)."""
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(early_stop=False)
    intercept_0 = solver.getIntercept(step=0)
    assert isinstance(intercept_0, float)
    assert np.isfinite(intercept_0)


@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_beta_path_first_col_is_zero(solver_data, factory):
    """betaPath[:, 0] must be all-zeros (the initial null-model state)."""
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.executeStep(early_stop=False)
    path = solver.getBetaPath()
    assert np.allclose(path[:, 0], 0.0), (
        f"{factory.__name__}: betaPath[:, 0] is not zero: {path[:, 0]}"
    )


# ---------------------------------------------------------------------------
# Tie-breaking seed (reproducibility API)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", STANDARD_SOLVER_FACTORIES,
                         ids=[f.__name__ for f in STANDARD_SOLVER_FACTORIES])
def test_set_tie_seed(solver_data, factory):
    """setTieSeed must be exposed on every solver and not interfere with execution."""
    X, D, y, n, p = solver_data
    solver = factory(X, D, y)
    solver.setTieSeed(42)
    solver.executeStep(2, True)
    assert solver.getNumSteps() > 0


# ---------------------------------------------------------------------------
# TENETAug diagnostics (regression: inherited getters must mirror the inner solver)
# ---------------------------------------------------------------------------

def test_tenet_aug_diagnostics_reflect_inner_solver(solver_data):
    """TENETAug historically returned empty diagnostics (getRSS() == [],
    getNumSteps() == 0) because its inherited base state was never populated."""
    X, D, y, n, p = solver_data
    solver = make_tenet_aug(X, D, y)
    solver.setTieSeed(7)
    solver.executeStep(2, True)

    assert solver.getNumSteps() > 0
    rss = solver.getRSS()
    assert len(rss) == solver.getNumSteps() + 1
    assert rss[0] > rss[-1]
    assert len(solver.getActives()) > 0
    assert len(solver.getActions()) == solver.getNumSteps()
    assert solver.getDummyStartIndex() == p
    assert solver.isConnected()

    # Exactly T_stop dummies must have entered the active set
    active_dummies = [j for j in solver.getActives() if j >= p]
    assert len(active_dummies) == 2


# ---------------------------------------------------------------------------
# Family getters
# ---------------------------------------------------------------------------

def test_tienet_and_tcdienet_group_getters(solver_data):
    X, D, y, n, p = solver_data
    groups = _groups_for(p)
    for cls in (TIENET_Solver, TCDIENET_Solver):
        s = cls(X, D, y, groups, SolverOptions(lambda2=0.5))
        s.executeStep(3)
        assert s.getLambda2() == pytest.approx(0.5)
        assert (np.asarray(s.getGroups()) == groups).all()
        assert s.getNumGroups() == len(np.unique(groups))
        assert s.getNumSteps() > 0
        del s   # a writer holds X and D until it is gone (in-place contract)


def test_cd_family_certified_getters(solver_data):
    X, D, y, n, p = solver_data
    s = TCDENET_Solver(X, D, y, SolverOptions(lambda2=0.5))
    s.executeStep(3)
    assert s.isCertifiedLocalization() is True
    assert isinstance(s.isCertifiedDegraded(), bool)
    assert s.getLambda2() == pytest.approx(0.5)
    assert len(s.getLambda()) == s.getNumSteps()   # one lambda_1 per recorded step
    assert all(g >= 0.0 for g in s.getRecordGaps())
    assert all(v >= 0.0 for v in s.getRecordKktViolations())
    assert s.getKktViolation() >= 0.0
    del s   # release the writer's claim on X and D before the next writer

    legacy = TCDLASSO_Solver(X, D, y, SolverOptions(cd_certified_localization=False))
    assert legacy.isCertifiedLocalization() is False
    assert legacy.getLambda2() == 0.0


def test_tikhonov_solvers_take_a_dense_k(solver_data):
    X, D, y, n, p = solver_data
    K = _group_mean_k(p)
    for cls in (TIKNET_Solver, TCDTIKNET_Solver):
        s = cls(X, D, y, K, SolverOptions(lambda2=0.5, fold_dummies=True))
        s.executeStep(3)
        assert s.getLambda2() == pytest.approx(0.5)
        if cls is not TCDTIKNET_Solver:   # the CD solver keeps K internal
            assert np.allclose(s.getTikhonovK(), K)
        assert s.getNumSteps() > 0
        del s   # a writer holds X and D until it is gone (in-place contract)
    tik = TCDTIKNET_Solver(X, D, y, K, SolverOptions(lambda2=0.5, fold_dummies=True))
    assert tik.isFoldedDummyCoupling() is True
    del tik
    with pytest.raises(ValueError):
        TIKNET_Solver(X, D, y, -np.eye(p), SolverOptions(lambda2=0.5))
