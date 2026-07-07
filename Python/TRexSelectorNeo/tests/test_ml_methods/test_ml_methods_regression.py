"""
Tests for scalers (ZScoreScaler, LpNormScaler) and ridge regression classes.

The *_inplace methods are true zero-copy in-place operations: they require a
Fortran-ordered float64 array and mutate it directly (no return value).
"""
import numpy as np
import pytest
from trex_selector_neo.ml_methods import (
    ZScoreScaler,
    LpNormScaler,
    NormType,
    RidgeCV,
    ElasticNet,
    ElasticNetCV,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data():
    rng = np.random.default_rng(10)
    n, p = 60, 8
    X = np.asfortranarray(rng.standard_normal((n, p)), dtype=np.float64)
    y = (X[:, 0] * 2.0 + X[:, 1] * -1.5 + rng.standard_normal(n)).astype(np.float64)
    return X.copy(order="F"), y, n, p


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


def test_zscore_scaler_get_centers(regression_data):
    X, y, n, p = regression_data
    scaler = ZScoreScaler()
    scaler.fit(X)
    centers = scaler.get_centers()
    assert isinstance(centers, np.ndarray)
    assert centers.shape == (p,)
    assert np.allclose(centers, X.mean(axis=0), atol=1e-10)


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
    X_scaled = X.copy(order="F")
    scaler.transform_inplace(X_scaled)
    col_means = X_scaled.mean(axis=0)
    assert np.allclose(col_means, 0.0, atol=1e-10)


def test_zscore_transform_unit_std(regression_data):
    X, y, n, p = regression_data
    scaler = ZScoreScaler()
    scaler.fit(X)
    X_scaled = X.copy(order="F")
    scaler.transform_inplace(X_scaled)
    col_stds = X_scaled.std(axis=0, ddof=1)
    assert np.allclose(col_stds, 1.0, atol=1e-6)


def test_zscore_transform_is_truly_inplace(regression_data):
    """transform_inplace mutates the passed Fortran-ordered array directly."""
    X, y, n, p = regression_data
    scaler = ZScoreScaler()
    scaler.fit(X)
    X_scaled = X.copy(order="F")
    original = X_scaled.copy(order="F")
    result = scaler.transform_inplace(X_scaled)
    assert result is None
    assert not np.allclose(X_scaled, original)


def test_zscore_inverse_transform_round_trip(regression_data):
    X, y, n, p = regression_data
    scaler = ZScoreScaler()
    scaler.fit(X)
    X_scaled = X.copy(order="F")
    scaler.transform_inplace(X_scaled)
    scaler.inverse_transform_inplace(X_scaled)
    assert np.allclose(X_scaled, X, atol=1e-12)


def test_zscore_fit_transform_inplace(regression_data):
    """fit_transform_inplace equals fit() + transform_inplace() (R's scale())."""
    X, y, n, p = regression_data
    X_one = X.copy(order="F")
    X_two = X.copy(order="F")

    one_call = ZScoreScaler()
    result = one_call.fit_transform_inplace(X_one)
    assert result is one_call

    two_calls = ZScoreScaler()
    two_calls.fit(X_two)
    two_calls.transform_inplace(X_two)

    assert np.allclose(X_one, X_two, atol=0.0)


# ---------------------------------------------------------------------------
# Returning (non-inplace) scaler API — layout-agnostic, leaves input untouched
# ---------------------------------------------------------------------------

def test_zscore_transform_returns_and_accepts_c_order(regression_data):
    """transform() returns a new array and accepts a C-ordered input (the
    natural 'copy then transform' idiom), without mutating the caller's array."""
    X, y, n, p = regression_data
    scaler = ZScoreScaler().fit(X)

    X_c = np.ascontiguousarray(X)          # C-ordered — would break *_inplace
    original = X_c.copy()
    out = scaler.transform(X_c)

    assert out is not None
    assert out.shape == X.shape
    assert np.array_equal(X_c, original), "transform() must not mutate its input"
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-10)


def test_zscore_fit_transform_returns_matches_inplace(regression_data):
    """The returning fit_transform() equals the in-place path numerically."""
    X, y, n, p = regression_data
    out = ZScoreScaler().fit_transform(np.ascontiguousarray(X))

    X_ref = X.copy(order="F")
    ZScoreScaler().fit_transform_inplace(X_ref)
    assert np.allclose(out, X_ref, atol=1e-12)


def test_zscore_transform_inverse_round_trip_returning(regression_data):
    X, y, n, p = regression_data
    scaler = ZScoreScaler().fit(X)
    out = scaler.transform(np.ascontiguousarray(X))
    back = scaler.inverse_transform(out)
    assert np.allclose(back, X, atol=1e-12)


def test_lpnorm_transform_returns_and_accepts_c_order(regression_data):
    X, y, n, p = regression_data
    scaler = LpNormScaler(NormType.L2).fit(X)
    X_c = np.ascontiguousarray(X)
    original = X_c.copy()
    out = scaler.transform(X_c)
    assert out.shape == X.shape
    assert np.array_equal(X_c, original)


def test_zscore_no_centering_uses_rms():
    """center=False, scale=True divides by the RMS around 0, as in R's scale()."""
    rng = np.random.default_rng(99)
    X = np.asfortranarray(rng.standard_normal((30, 4)) + 5.0, dtype=np.float64)
    scaler = ZScoreScaler(center=False, scale=True)
    scaler.fit(X)
    # Column means should still be non-zero after transform (not centered)
    X_copy = X.copy(order="F")
    scaler.transform_inplace(X_copy)
    assert not np.allclose(X_copy.mean(axis=0), 0.0, atol=0.5)
    # Scale = root-mean-square around 0 with Bessel denominator (R behavior)
    rms = np.sqrt((X**2).sum(axis=0) / (X.shape[0] - 1))
    assert np.allclose(scaler.get_scales(), rms, atol=1e-12)


def test_zscore_dropped_indices_constant_column():
    """Constant columns are flagged as dropped but still centered (sklearn)."""
    X = np.asfortranarray(np.ones((20, 3)), dtype=np.float64)
    X[:, 0] = np.arange(20, dtype=np.float64)  # non-constant column 0
    scaler = ZScoreScaler()
    scaler.fit(X)
    dropped = scaler.get_dropped_indices()
    assert 1 in dropped and 2 in dropped  # columns 1 and 2 are constant
    # sklearn semantics: degenerate columns get scale 1 and are still centered
    scaler.transform_inplace(X)
    assert np.allclose(X[:, 1], 0.0, atol=0.0)
    assert np.allclose(X[:, 2], 0.0, atol=0.0)


# ---------------------------------------------------------------------------
# LpNormScaler
# ---------------------------------------------------------------------------

def test_lpnorm_scaler_l2(regression_data):
    X, y, n, p = regression_data
    scaler = LpNormScaler(NormType.L2)
    scaler.fit(X)
    X_scaled = X.copy(order="F")
    scaler.transform_inplace(X_scaled)
    # Each column should have L2 norm ≈ 1
    col_norms = np.linalg.norm(X_scaled, axis=0)
    assert np.allclose(col_norms, 1.0, atol=1e-6)


def test_lpnorm_scaler_l1(regression_data):
    X, y, n, p = regression_data
    scaler = LpNormScaler(NormType.L1)
    scaler.fit(X)
    X_scaled = X.copy(order="F")
    scaler.transform_inplace(X_scaled)
    # Each column should have L1 norm ≈ 1
    col_norms = np.sum(np.abs(X_scaled), axis=0)
    assert np.allclose(col_norms, 1.0, atol=1e-6)


def test_lpnorm_center_only(regression_data):
    """center=True, scale=False only removes the column means (SPCA usage)."""
    X, y, n, p = regression_data
    scaler = LpNormScaler(NormType.L2, center=True, scale=False)
    X_scaled = X.copy(order="F")
    scaler.fit_transform_inplace(X_scaled)
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(X_scaled, X - X.mean(axis=0), atol=1e-12)
    assert np.allclose(scaler.get_scales(), 1.0, atol=0.0)


def test_lpnorm_scaler_not_fitted_initially():
    scaler = LpNormScaler(NormType.L2)
    assert not scaler.is_fitted()


def test_lpnorm_fit_returns_self(regression_data):
    X, y, n, p = regression_data
    scaler = LpNormScaler(NormType.L2)
    result = scaler.fit(X)
    assert result is scaler
    assert scaler.is_fitted()


# ---------------------------------------------------------------------------
# Model selection: RidgeCV
# ---------------------------------------------------------------------------

def test_ridge_cv_fit_returns_self_and_selects_lambda(regression_data):
    X, y, n, p = regression_data
    cv = RidgeCV()
    assert cv.fit(X, y, n_folds=5, n_lambda=100, seed=0) is cv
    lam = np.asarray(cv.lambdas())
    assert lam.ndim == 1 and len(lam) == 100
    # ridge_cv_svd uses an ASCENDING grid (low -> high lambda)
    assert np.all(np.diff(lam) >= 0), "ridge lambda grid is ascending"
    # 1se rule selects a lambda at least as regularized (larger) than the min,
    # which in an ascending grid means index_1se >= index_min
    assert cv.cv_1se() >= cv.cv_min()
    assert 0 <= cv.index_min() < len(lam)
    assert cv.index_min() <= cv.index_1se() < len(lam)
    assert len(np.asarray(cv.cv_mse())) == len(lam)
    assert len(np.asarray(cv.cv_sem())) == len(lam)


def test_ridge_cv_deterministic_under_seed(regression_data):
    X, y, n, p = regression_data
    a = RidgeCV().fit(X, y, n_folds=5, n_lambda=50, seed=7)
    b = RidgeCV().fit(X, y, n_folds=5, n_lambda=50, seed=7)
    assert a.cv_min() == b.cv_min()
    assert np.allclose(np.asarray(a.cv_mse()), np.asarray(b.cv_mse()))


# ---------------------------------------------------------------------------
# Model selection: ElasticNet (coordinate-descent path)
# ---------------------------------------------------------------------------

def test_elasticnet_path_shapes_and_recovery(regression_data):
    X, y, n, p = regression_data
    en = ElasticNet()
    assert en.fit(X, y, alpha=1.0, n_lambda=60) is en
    coef = np.asarray(en.coef())
    lam = np.asarray(en.lambdas())
    assert coef.shape == (p, len(lam))
    assert en.converged()
    # deviance ratio is non-decreasing along a descending lambda path
    dev = np.asarray(en.dev_ratio())
    assert np.all(np.diff(dev) >= -1e-8)
    # predictions align with the path
    pred = np.asarray(en.predict(X))
    assert pred.shape == (n, len(lam))
    # true signal (cols 0,1) is active at the least-regularized end
    assert np.any(np.abs(coef[:2, -1]) > 1e-3)


def test_elasticnet_fit_grid_matches_auto_path(regression_data):
    X, y, n, p = regression_data
    auto = ElasticNet().fit(X, y, alpha=0.5, n_lambda=40)
    grid = ElasticNet().fit_grid(X, y, auto.lambdas(), alpha=0.5)
    assert np.allclose(np.asarray(grid.coef()), np.asarray(auto.coef()), atol=1e-10)


def test_elasticnet_ridge_limit_is_dense(regression_data):
    """alpha=0 (ridge) leaves all coefficients non-zero."""
    X, y, n, p = regression_data
    en = ElasticNet().fit(X, y, alpha=0.0, n_lambda=30)
    coef_last = np.asarray(en.coef())[:, -1]
    assert np.all(np.abs(coef_last) > 0.0)


# ---------------------------------------------------------------------------
# Model selection: ElasticNetCV
# ---------------------------------------------------------------------------

def test_elasticnet_cv_selects_lambda(regression_data):
    X, y, n, p = regression_data
    cv = ElasticNetCV()
    assert cv.fit(X, y, alpha=0.5, n_folds=5, n_lambda=100, seed=0) is cv
    lam = np.asarray(cv.lambdas())
    assert np.all(np.diff(lam) <= 0)
    assert cv.cv_1se() >= cv.cv_min()
    assert 0 <= cv.index_1se() <= cv.index_min() < len(lam)
    assert len(np.asarray(cv.cv_mse())) == len(lam)


def test_elasticnet_cv_deterministic_under_seed(regression_data):
    X, y, n, p = regression_data
    a = ElasticNetCV().fit(X, y, alpha=0.5, n_folds=5, n_lambda=50, seed=3)
    b = ElasticNetCV().fit(X, y, alpha=0.5, n_folds=5, n_lambda=50, seed=3)
    assert a.cv_min() == b.cv_min()
    assert np.allclose(np.asarray(a.cv_mse()), np.asarray(b.cv_mse()))
