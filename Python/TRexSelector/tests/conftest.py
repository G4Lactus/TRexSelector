"""
Shared pytest fixtures for all TRexSelector test modules.
"""
import numpy as np
import pytest
import trex_selector


@pytest.fixture
def signal_data():
    rng = np.random.default_rng(42)
    n, p = 50, 20
    X = rng.standard_normal((n, p))
    y = X[:, 0] * 5 + X[:, 1] * -4 + X[:, 2] * 3 + rng.standard_normal(n)
    return X, y, n, p


@pytest.fixture
def fast_control():
    ctrl = trex_selector.TRexControlParameter()
    ctrl.K = 3
    return ctrl


@pytest.fixture
def solver_data():
    """Small dataset for solver unit tests (n=50, p=5 real + p=5 dummies)."""
    rng_X = np.random.default_rng(0)
    rng_D = np.random.default_rng(1)
    n, p = 50, 5
    X = np.asfortranarray(rng_X.standard_normal((n, p)).astype(np.float64))
    D = np.asfortranarray(rng_D.standard_normal((n, p)).astype(np.float64))
    y = (X[:, 0] * 3.0 + X[:, 1] * -2.0 + rng_X.standard_normal(n)).astype(np.float64)
    return X, D, y, n, p


@pytest.fixture
def cluster_data():
    """Small dataset for clustering tests (n=20, p=6)."""
    rng = np.random.default_rng(2)
    n, p = 20, 6
    X = rng.standard_normal((n, p)).astype(np.float64)
    return X, n, p
