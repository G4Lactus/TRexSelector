"""getMemoryBytes(): the heap bytes a T-solver holds (the C++ memoryBytes()).

X and D are borrowed and not counted. Mirrors
cpp/tests/test_tsolvers/ad/test_tsolvers_memory.cpp and the R test
test-tsolvers-memory.R (same deterministic data, same TLARS count).
"""
import numpy as np
import pytest

from trexselector.tsolvers.ad import SolverOptions
from trexselector.tsolvers.ad.afs_based import TAFS_Solver
from trexselector.tsolvers.ad.cd_based import (
    TCDENET_Solver,
    TCDIENET_Solver,
    TCDLASSO_Solver,
    TCDTIKNET_Solver,
)
from trexselector.tsolvers.ad.lars_based import (
    TENET_Solver,
    TENETAug_Solver,
    TIENET_Solver,
    TIENETAug_Solver,
    TIKNET_Solver,
    TLARS_Solver,
    TLASSO_Solver,
    TSTAGEWISE_Solver,
    TSTEPWISE_Solver,
)
from trexselector.tsolvers.ad.omp_based import (
    TACGP_Solver,
    TGP_Solver,
    TMP_Solver,
    TNCGMP_Solver,
    TOMP_Solver,
    TOOLS_Solver,
)

N, P_X, P_D = 20, 10, 10


def _reader(**kwargs):
    return SolverOptions(normalize=False, intercept=False, **kwargs)


@pytest.fixture
def deterministic():
    """X[i, j] = sin(i + 2 j), D[i, j] = cos(i + 3 j) (1-based), y = cos(i)."""
    i = np.arange(1, N + 1)[:, None]
    X = np.asfortranarray(np.sin(i + 2 * np.arange(1, P_X + 1)[None, :]))
    D = np.asfortranarray(np.cos(i + 3 * np.arange(1, P_D + 1)[None, :]))
    y = np.cos(np.arange(1, N + 1)).astype(np.float64)
    return X, D, y


@pytest.fixture
def gaussian():
    rng = np.random.default_rng(3)
    n = 40
    X = np.asfortranarray(rng.standard_normal((n, P_X)))
    D = np.asfortranarray(rng.standard_normal((n, P_D)))
    y = X[:, 0] * 2.0 - X[:, 1] + rng.standard_normal(n)
    return X, D, y


_GROUPS = np.repeat(np.arange(P_X // 2, dtype=np.int32), 2)
_K = np.eye(P_X)

FACTORIES = {
    "TLARS": lambda X, D, y: TLARS_Solver(X, D, y, _reader()),
    "TLASSO": lambda X, D, y: TLASSO_Solver(X, D, y, _reader()),
    "TSTEPWISE": lambda X, D, y: TSTEPWISE_Solver(X, D, y, _reader()),
    "TSTAGEWISE": lambda X, D, y: TSTAGEWISE_Solver(X, D, y, _reader()),
    "TENET": lambda X, D, y: TENET_Solver(X, D, y, _reader(lambda2=0.5)),
    "TENETAug": lambda X, D, y: TENETAug_Solver(X, D, y, _reader(lambda2=0.5)),
    "TIENET": lambda X, D, y: TIENET_Solver(X, D, y, _GROUPS, _reader(lambda2=0.5)),
    "TIENETAug": lambda X, D, y: TIENETAug_Solver(X, D, y, _GROUPS, _reader(lambda2=0.5)),
    "TIKNET": lambda X, D, y: TIKNET_Solver(X, D, y, _K, _reader(lambda2=0.5)),
    "TOMP": lambda X, D, y: TOMP_Solver(X, D, y, _reader()),
    "TGP": lambda X, D, y: TGP_Solver(X, D, y, _reader()),
    "TACGP": lambda X, D, y: TACGP_Solver(X, D, y, _reader()),
    "TMP": lambda X, D, y: TMP_Solver(X, D, y, _reader()),
    "TNCGMP": lambda X, D, y: TNCGMP_Solver(X, D, y, _reader()),
    "TOOLS": lambda X, D, y: TOOLS_Solver(X, D, y, _reader()),
    "TAFS": lambda X, D, y: TAFS_Solver(X, D, y, _reader(rho_afs=0.5)),
    "TCDLASSO": lambda X, D, y: TCDLASSO_Solver(X, D, y, _reader()),
    "TCDENET": lambda X, D, y: TCDENET_Solver(X, D, y, _reader(lambda2=0.5)),
    "TCDIENET": lambda X, D, y: TCDIENET_Solver(X, D, y, _GROUPS, _reader(lambda2=0.5)),
    "TCDTIKNET": lambda X, D, y: TCDTIKNET_Solver(X, D, y, _K, _reader(lambda2=0.5)),
}


@pytest.mark.parametrize("name", list(FACTORIES))
def test_positive_int_that_grows_with_the_path(gaussian, name):
    X, D, y = gaussian
    solver = FACTORIES[name](X, D, y)
    before = solver.getMemoryBytes()
    assert isinstance(before, int)
    assert before > 0
    solver.executeStep(3)
    assert solver.getNumSteps() > 0
    assert solver.getMemoryBytes() > before


def test_tlars_construction_formula(deterministic):
    X, D, y = deterministic
    solver = TLARS_Solver(X, D, y, _reader())
    P = P_X + P_D
    M = 8 * min(P, N)  # step limit 8 min(max_actives, effective_n)
    # y copy and residual (n), correlations, inactive list and Kahan
    # compensation (P each), one path step (48 bytes), one RSS, R2 and DoF
    # entry, dummy record and lambda sequence reserved for M + 1 steps
    expected = 16 * N + 24 * P + 48 + 24 + 16 * (M + 1)
    assert solver.getMemoryBytes() == expected == 3448


def test_doubling_n_moves_only_the_n_vectors(gaussian):
    X, D, y = gaussian
    small = TLARS_Solver(X, D, y, _reader()).getMemoryBytes()
    X2 = np.asfortranarray(np.vstack([X, X]))
    D2 = np.asfortranarray(np.vstack([D, D]))
    y2 = np.concatenate([y, y])
    large = TLARS_Solver(X2, D2, y2, _reader()).getMemoryBytes()
    assert large - small == 2 * 8 * X.shape[0]


def test_tenet_aug_counts_its_owned_copies(deterministic):
    X, D, y = deterministic
    solver = TENETAug_Solver(X, D, y, _reader(lambda2=0.5))
    P = P_X + P_D
    assert solver.getMemoryBytes() > 8 * (N + P) * (P + 1)
