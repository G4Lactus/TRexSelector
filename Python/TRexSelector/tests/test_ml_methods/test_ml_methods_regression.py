"""
Tests for scalers (ZScoreScaler, LpNormScaler) and ridge regression classes.
"""
import numpy as np
import pytest
from trex_selector.ml_methods import (
    ZScoreScaler,
    LpNormScaler,
    NormType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data():
    rng = np.random.default_rng(10)
    n, p = 60, 8
    X = rng.standard_normal((n, p)).astype(np.float64)
    y = (X[:, 0] * 2.0 + X[:, 1] * -1.5 + rng.standard_normal(n)).astype(np.float64)
    return X.copy(), y, n, p


# ---------------------------------------------------------------------------
# ZScoreScaler
# ---------------------------------------------------------------------------

def test_zscore_scaler_not_fitted_initially():
    scaler = ZScoreScaler()
    assert not scaler.is_fitted()


def test_zscore_scaler_fit_returns_self(regression_data):
    X, y, n, p = regression_data
    scaler = ZScoreScaler()
    result = scaler.fit(X)
    assert result is scaler
    assert scaler.is_fitted()


def test_zscore_scaler_get_means(regression_data):
    X, y, n, p = regression_data
    scaler = ZScoreScaler()
    scaler.fit(X)
    means = scaler.get_means()
    assert isinstance(means, np.ndarray)
    assert means.shape == (p,)
    assert np.allclose(means, X.mean(axis=0), atol=1e-10)


def test_zscore_scaler_get_scales(regression_data):
    X, y, n, p = regression_data
    scaler = ZScoreScaler()
    scaler.fit(X)
    scales = scaler.get_scales()
    assert isinstance(scales, np.ndarray)
    assert scales.shape == (p,)
    assert np.all(scales > 0)


def test_zscore_transform_zero_mean(regression_data):
    X, y, n, p = regression_data
    scaler = ZScoreScaler()
    scaler.fit(X)
    X_scaled = X.copy()
    X_scaled = scaler.transform_inplace(X_scaled)
    col_means = X_scaled.mean(axis=0)
    assert np.allclose(col_means, 0.0, atol=1e-10)


def test_zscore_transform_unit_std(regression_data):
    X, y, n, p = regression_data
    scaler = ZScoreScaler()
    scaler.fit(X)
    X_scaled = X.copy()
    X_scaled = scaler.transform_inplace(X_scaled)
    col_stds = X_scaled.std(axis=0, ddof=1)
    assert np.allclose(col_stds, 1.0, atol=1e-6)


def test_zscore_no_mean_centering():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((30, 4)).astype(np.float64) + 5.0
    scaler = ZScoreScaler(with_mean=False, with_std=True)
    scaler.fit(X)
    X_copy = X.copy()
    scaler.transform_inplace(X_copy)
    # Column means should still be non-zero (not centered)
    assert not np.allclose(X_copy.mean(axis=0), 0.0, atol=0.5)


def test_zscore_dropped_indices_constant_column():
    """Constant columns should be flagged as dropped."""
    X = np.ones((20, 3), dtype=np.float64)
    X[:, 0] = np.arange(20, dtype=np.float64)  # non-constant column 0
    scaler = ZScoreScaler()
    scaler.fit(X)
    dropped = scaler.get_dropped_indices()
    assert 1 in dropped or 2 in dropped  # columns 1 and 2 are constant


# ---------------------------------------------------------------------------
# LpNormScaler
# ---------------------------------------------------------------------------

def test_lpnorm_scaler_l2(regression_data):
    X, y, n, p = regression_data
    scaler = LpNormScaler(NormType.L2)
    scaler.fit(X)
    X_scaled = X.copy()
    X_scaled = scaler.transform_inplace(X_scaled)
    # Each column should have L2 norm ≈ 1
    col_norms = np.linalg.norm(X_scaled, axis=0)
    assert np.allclose(col_norms, 1.0, atol=1e-6)


def test_lpnorm_scaler_l1(regression_data):
    X, y, n, p = regression_data
    scaler = LpNormScaler(NormType.L1)
    scaler.fit(X)
    X_scaled = X.copy()
    X_scaled = scaler.transform_inplace(X_scaled)
    # Each column should have L1 norm ≈ 1
    col_norms = np.sum(np.abs(X_scaled), axis=0)
    assert np.allclose(col_norms, 1.0, atol=1e-6)


def test_lpnorm_scaler_not_fitted_initially():
    scaler = LpNormScaler(NormType.L2)
    assert not scaler.is_fitted()


def test_lpnorm_fit_returns_self(regression_data):
    X, y, n, p = regression_data
    scaler = LpNormScaler(NormType.L2)
    result = scaler.fit(X)
    assert result is scaler
    assert scaler.is_fitted()
