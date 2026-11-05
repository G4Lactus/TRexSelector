"""The raw-X (low_memory) elastic-net engine against the copying engine:
same estimator, same grid, agreement to rounding; the caller's array is
never written."""

import numpy as np
import pytest

from trexselector.ml_methods import ElasticNet, ElasticNetCV


@pytest.fixture
def data():
    rng = np.random.default_rng(83)
    n, p = 80, 30
    X = np.asfortranarray(rng.standard_normal((n, p)))
    X[:, 4] += 500.0                                   # a large-mean column
    y = X[:, 0] * 2.0 - X[:, 4] * 0.01 + rng.standard_normal(n)
    return X, y


def test_low_memory_path_agrees(data):
    X, y = data
    X0 = X.copy(order="F")
    a = ElasticNet()
    a.fit(X, y, 0.7, 30, -1.0, True, True, False, 100000, 1e-9)
    b = ElasticNet(low_memory=True)
    assert b.low_memory() is True
    b.fit(X, y, 0.7, 30, -1.0, True, True, False, 100000, 1e-9)
    assert np.array_equal(X, X0)                       # never written
    np.testing.assert_allclose(b.lambdas(), a.lambdas(), rtol=1e-12)
    scale = np.abs(a.coef()).max()
    np.testing.assert_allclose(b.coef(), a.coef(), atol=1e-7 * scale, rtol=0)
    np.testing.assert_allclose(b.intercepts(), a.intercepts(), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(b.predict(np.asfortranarray(X[:5])),
                               a.predict(np.asfortranarray(X[:5])), rtol=1e-7, atol=1e-9)


def test_low_memory_cv_agrees(data):
    X, y = data
    a = ElasticNetCV()
    a.fit(X, y, 0.5, 5, 30, -1.0, 3, True, True, 100000, 1e-9)
    b = ElasticNetCV(low_memory=True)
    b.fit(X, y, 0.5, 5, 30, -1.0, 3, True, True, 100000, 1e-9)
    np.testing.assert_allclose(b.cv_mse(), a.cv_mse(), rtol=1e-7)
    assert b.index_min() == a.index_min()
    assert b.index_1se() == a.index_1se()


def test_low_memory_reads_a_readonly_array(data):
    X, y = data
    Xro = X.copy(order="F")
    Xro.setflags(write=False)
    b = ElasticNet(low_memory=True)
    b.fit(Xro, y, 1.0, 20)                             # a read-only view is enough
    assert b.coef().shape[0] == X.shape[1]


def test_ridge_cv_low_memory_agrees():
    from trexselector.ml_methods import RidgeCV
    rng = np.random.default_rng(84)
    n, p = 60, 300
    X = np.asfortranarray(rng.standard_normal((n, p)))
    y = X[:, 0] * 2.0 + rng.standard_normal(n)
    a = RidgeCV()
    a.fit(X, y, 5, 40, 1000.0, 4)
    b = RidgeCV(low_memory=True)
    assert b.low_memory() is True
    b.fit(X, y, 5, 40, 1000.0, 4)
    np.testing.assert_allclose(b.lambdas(), a.lambdas(), rtol=1e-14)
    np.testing.assert_allclose(b.cv_mse(), a.cv_mse(), rtol=1e-7)
    assert b.index_min() == a.index_min()
    assert b.index_1se() == a.index_1se()


def test_svd_thresholds_select_randomized_path():
    from trexselector.ml_methods import SVDSolver
    rng = np.random.default_rng(85)
    # a decaying spectrum (rank-4 signal above noise): the randomized route's regime
    A = rng.standard_normal((120, 4)) * np.array([40.0, 30.0, 20.0, 10.0])
    X = np.asfortranarray(A @ rng.standard_normal((4, 80)) + 0.1 * rng.standard_normal((120, 80)))
    direct = SVDSolver().compute(X, 4)
    rnd = SVDSolver(min_dim_thresh=16, block_size=32).compute(X, 4)
    assert rnd.U.shape == (120, 4) and rnd.V.shape == (80, 4)
    np.testing.assert_allclose(rnd.U.T @ rnd.U, np.eye(4), atol=1e-10)
    # the smallest retained component is the least accurate one of the
    # randomized route (10 oversamples, no power iteration)
    np.testing.assert_allclose(rnd.S, direct.S, rtol=1e-4)


def test_cv_diagnostics_are_exposed():
    from trexselector.ml_methods import ElasticNetCV, RidgeCV
    rng = np.random.default_rng(87)
    X = np.asfortranarray(rng.standard_normal((60, 20)))
    y = X[:, 0] * 2.0 + rng.standard_normal(60)
    r = RidgeCV()
    r.fit(X, y, 5, 30, 1000.0, 1)
    assert isinstance(r.interpolates(), bool)
    assert isinstance(r.selection_at_grid_edge(), bool)
    e = ElasticNetCV()
    e.fit(X, y, 0.5, 5, 30, -1.0, 1, True, True, 100000, 1e-7)
    assert isinstance(e.selection_at_grid_edge(), bool)
    assert isinstance(e.grid_truncated(), bool)
