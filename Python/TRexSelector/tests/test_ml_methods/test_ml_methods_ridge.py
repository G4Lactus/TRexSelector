"""
Tests for RidgeSolver (trex_selector_neo.ml_methods.RidgeSolver).
"""
import numpy as np
import pytest
from trex_selector_neo.ml_methods import RidgeSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tall_regression():
    rng = np.random.default_rng(12)
    n, p = 20, 5
    X = np.array([[np.sin(i * j) for j in range(1, p + 1)] for i in range(1, n + 1)],
                 dtype=np.float64)
    beta_true = np.array([1.0, -2.0, 0.0, 0.5, -1.0])
    y = X @ beta_true + rng.standard_normal(n) * 0.01
    return X, y, n, p


@pytest.fixture
def wide_regression():
    rng = np.random.default_rng(13)
    n, p = 8, 15
    X = np.array([[np.cos(i * j) for j in range(1, p + 1)] for i in range(1, n + 1)],
                 dtype=np.float64)
    y = rng.standard_normal(n).astype(np.float64)
    return X, y, n, p


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_ridge_solve_output_length_tall(tall_regression):
    X, y, n, p = tall_regression
    beta = RidgeSolver.solve(X, y, lambda_val=0.1)
    assert beta.shape == (p,)


def test_ridge_solve_output_length_wide(wide_regression):
    X, y, n, p = wide_regression
    beta = RidgeSolver.solve(X, y, lambda_val=1.0)
    assert beta.shape == (p,)


# ---------------------------------------------------------------------------
# KKT conditions
# ---------------------------------------------------------------------------

def test_ridge_kkt_primal(tall_regression):
    """(X'X + lambda I) beta == X' y"""
    X, y, _, _ = tall_regression
    lam = 0.5
    beta = RidgeSolver.solve(X, y, lambda_val=lam)
    lhs = X.T @ X @ beta + lam * beta
    rhs = X.T @ y
    assert np.allclose(lhs, rhs, atol=1e-8)


def test_ridge_kkt_dual(wide_regression):
    """(X'X + lambda I) beta == X' y  (dual Cholesky path)"""
    X, y, _, _ = wide_regression
    lam = 1.0
    beta = RidgeSolver.solve(X, y, lambda_val=lam)
    lhs = X.T @ X @ beta + lam * beta
    rhs = X.T @ y
    assert np.allclose(lhs, rhs, atol=1e-8)


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------

def test_ridge_shrinkage_increases_with_lambda():
    rng = np.random.default_rng(14)
    n, p = 30, 5
    X = rng.standard_normal((n, p)).astype(np.float64)
    y = X @ np.ones(p) + rng.standard_normal(n) * 0.1
    norm_small = np.linalg.norm(RidgeSolver.solve(X, y, lambda_val=1e-4))
    norm_large = np.linalg.norm(RidgeSolver.solve(X, y, lambda_val=1e4))
    assert norm_small > norm_large


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_ridge_solve_deterministic(tall_regression):
    X, y, _, _ = tall_regression
    b1 = RidgeSolver.solve(X, y, lambda_val=0.1)
    b2 = RidgeSolver.solve(X, y, lambda_val=0.1)
    assert np.allclose(b1, b2)
