"""
Tests for PCA (trex_selector.ml_methods.PCA).

Note on memory layout
---------------------
The C++ PCA::fit() takes an Eigen::Ref<Eigen::MatrixXd>, which requires
Fortran-contiguous (column-major) memory. All fixtures that pass X to fit()
must therefore use np.asfortranarray(). This matches the convention used
throughout the TRex Python package (all selector wrappers call
np.asfortranarray internally).
"""
import numpy as np
import pytest
from trex_selector.ml_methods import PCA, PCAResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_30x8():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 8)).astype(np.float64)
    return np.asfortranarray(X)   # Eigen requires F-order


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

def test_pca_fit_result_dimensions(data_30x8):
    M = 3
    pca = PCA(center=True)
    res = pca.fit(data_30x8, M)
    assert isinstance(res, PCAResult)
    assert res.Z.shape == (30, M)
    assert res.V.shape == (8, M)
    assert res.explained_variance.shape == (M,)


def test_pca_loadings_orthonormal(data_30x8):
    M = 4
    pca = PCA()
    res = pca.fit(data_30x8, M)
    assert np.allclose(res.V.T @ res.V, np.eye(M), atol=1e-10)


def test_pca_getters_match_result(data_30x8):
    M = 3
    pca = PCA()
    res = pca.fit(data_30x8, M)
    assert np.allclose(pca.get_loadings(), res.V, atol=1e-12)
    assert np.allclose(pca.get_explained_variance(), res.explained_variance, atol=1e-12)


# ---------------------------------------------------------------------------
# Centering
# ---------------------------------------------------------------------------

def test_pca_stores_column_means(data_30x8):
    # Capture means before fit() centers X in-place
    col_means = data_30x8.mean(axis=0).copy()
    pca = PCA(center=True)
    pca.fit(data_30x8, 3)
    assert np.allclose(pca.get_mean(), col_means, atol=1e-12)


def test_pca_no_center_mean_is_zero(data_30x8):
    pca = PCA(center=False)
    pca.fit(data_30x8, 3)
    assert np.allclose(pca.get_mean(), np.zeros(8), atol=1e-12)


# ---------------------------------------------------------------------------
# Explained variance
# ---------------------------------------------------------------------------

def test_pca_explained_variance_non_negative(data_30x8):
    pca = PCA()
    res = pca.fit(data_30x8, 6)
    assert np.all(res.explained_variance >= 0)


def test_pca_explained_variance_non_increasing(data_30x8):
    pca = PCA()
    res = pca.fit(data_30x8, 6)
    assert np.all(np.diff(res.explained_variance) <= 1e-10)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def test_pca_transform_matches_fit_scores(data_30x8):
    """fit() centers X in-place; restore() then transform(X) reproduces fit scores."""
    M = 3
    pca = PCA(center=True)
    res = pca.fit(data_30x8, M)
    pca.restore()
    Z_transform = pca.transform(data_30x8)
    assert np.allclose(Z_transform, res.Z, atol=1e-10)


def test_pca_transform_held_out_shape():
    rng = np.random.default_rng(5)
    X     = np.asfortranarray(rng.standard_normal((30, 6)).astype(np.float64))
    X_new = rng.standard_normal((10, 6)).astype(np.float64)
    pca = PCA()
    pca.fit(X, 3)
    Z_new = pca.transform(X_new)
    assert Z_new.shape == (10, 3)


# ---------------------------------------------------------------------------
# RAII restore
# ---------------------------------------------------------------------------

def test_pca_restore_uncenters_X():
    """fit() modifies X in-place; restore() brings it back to original values."""
    rng = np.random.default_rng(7)
    X = np.asfortranarray(rng.standard_normal((20, 4)).astype(np.float64))
    X_orig = X.copy()   # deep copy before in-place centering
    pca = PCA(center=True)
    pca.fit(X, 2)
    pca.restore()
    assert np.allclose(X, X_orig, atol=1e-12)
