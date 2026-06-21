"""
Tests for SVDSolver (trex_selector.ml_methods.SVDSolver).
"""
import numpy as np
import pytest
from trex_selector.ml_methods import SVDSolver, SVDResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tall_matrix():
    rng = np.random.default_rng(42)
    return rng.standard_normal((40, 8)).astype(np.float64)


@pytest.fixture
def wide_matrix():
    rng = np.random.default_rng(7)
    return rng.standard_normal((10, 30)).astype(np.float64)


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

def test_svd_result_dimensions_tall(tall_matrix):
    M = 3
    res = SVDSolver().compute(tall_matrix, M)
    assert isinstance(res, SVDResult)
    assert res.U.shape == (40, M)
    assert res.S.shape == (M,)
    assert res.V.shape == (8, M)


def test_svd_result_dimensions_wide(wide_matrix):
    M = 4
    res = SVDSolver().compute(wide_matrix, M)
    assert res.U.shape == (10, M)
    assert res.S.shape == (M,)
    assert res.V.shape == (30, M)


# ---------------------------------------------------------------------------
# Singular values
# ---------------------------------------------------------------------------

def test_svd_singular_values_non_negative(tall_matrix):
    res = SVDSolver().compute(tall_matrix, 5)
    assert np.all(res.S >= 0)


def test_svd_singular_values_non_increasing(tall_matrix):
    res = SVDSolver().compute(tall_matrix, 5)
    assert np.all(np.diff(res.S) <= 0)


# ---------------------------------------------------------------------------
# Orthonormality
# ---------------------------------------------------------------------------

def test_svd_U_columns_orthonormal(tall_matrix):
    M = 4
    U = SVDSolver().compute(tall_matrix, M).U
    assert np.allclose(U.T @ U, np.eye(M), atol=1e-10)


def test_svd_V_columns_orthonormal(tall_matrix):
    M = 4
    V = SVDSolver().compute(tall_matrix, M).V
    assert np.allclose(V.T @ V, np.eye(M), atol=1e-10)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def test_svd_full_rank_reconstruction():
    rng = np.random.default_rng(9)
    n, p = 8, 6
    X = rng.standard_normal((n, p)).astype(np.float64)
    M = min(n, p)
    res = SVDSolver().compute(X, M)
    X_rec = res.U @ np.diag(res.S) @ res.V.T
    assert np.allclose(X_rec, X, atol=1e-8)


def test_svd_low_rank_residual_bounded():
    rng = np.random.default_rng(11)
    n, p = 30, 10
    X = rng.standard_normal((n, p)).astype(np.float64)
    M = 3
    res = SVDSolver().compute(X, M)
    # Residual norm <= sigma_{M+1} * sqrt(rank)
    res_full = SVDSolver().compute(X, p)
    sigma_next = res_full.S[M]
    residual = np.linalg.norm(X - res.U @ np.diag(res.S) @ res.V.T, 'fro')
    assert residual <= sigma_next * np.sqrt(min(n, p)) + 1e-8
