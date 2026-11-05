"""
Tests for agglomerative clustering and cut_tree.
"""
import numpy as np
import pytest
from trexselector.ml_methods.clustering import (
    LinkageMethod,
    DistanceMetric,
    agglomerative_cluster,
    cut_tree,
)


# ---------------------------------------------------------------------------
# Basic shape and monotonicity
# ---------------------------------------------------------------------------

def test_ward_returns_correct_shape(cluster_data):
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, LinkageMethod.Ward, DistanceMetric.Euclidean)
    assert isinstance(linkage, np.ndarray)
    assert linkage.shape == (n - 1, 4)


def test_linkage_column2_nondecreasing(cluster_data):
    """Column 2 (merge height/distance) must be non-decreasing."""
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, LinkageMethod.Ward, DistanceMetric.Euclidean)
    heights = linkage[:, 2]
    assert np.all(np.diff(heights) >= -1e-10)


@pytest.mark.parametrize("method", [
    LinkageMethod.Average,
    LinkageMethod.Complete,
    LinkageMethod.Single,
])
def test_euclidean_methods_correct_shape(cluster_data, method):
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, method, DistanceMetric.Euclidean)
    assert linkage.shape == (n - 1, 4)


@pytest.mark.parametrize("method", [
    LinkageMethod.Average,
    LinkageMethod.Complete,
    LinkageMethod.Single,
])
def test_correlation_metric_non_ward(cluster_data, method):
    """Non-Ward methods should accept Correlation distance."""
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, method, DistanceMetric.Correlation)
    assert linkage.shape == (n - 1, 4)


@pytest.mark.parametrize("method", [
    LinkageMethod.Average,
    LinkageMethod.Complete,
    LinkageMethod.Single,
])
def test_manhattan_metric_non_ward(cluster_data, method):
    """Non-Ward methods should accept Manhattan distance."""
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, method, DistanceMetric.Manhattan)
    assert linkage.shape == (n - 1, 4)


def test_ward_requires_euclidean(cluster_data):
    """Ward linkage with non-Euclidean distance must raise."""
    X, n, p = cluster_data
    with pytest.raises(Exception):
        agglomerative_cluster(X, LinkageMethod.Ward, DistanceMetric.Correlation)


def test_ward_lsh_approx_permitted(cluster_data):
    """Ward + Correlation_LSH_Approx is the supported exception: the core
    dispatcher runs Ward in a 64-D SimHash-projected Euclidean space."""
    X, n, p = cluster_data
    linkage = agglomerative_cluster(
        X, LinkageMethod.Ward, DistanceMetric.Correlation_LSH_Approx
    )
    assert linkage.shape == (n - 1, 4)


# ---------------------------------------------------------------------------
# Cluster indices from linkage matrix (column 0 & 1 in range)
# ---------------------------------------------------------------------------

def test_linkage_indices_valid(cluster_data):
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, LinkageMethod.Ward, DistanceMetric.Euclidean)
    # Each row merges two existing clusters; node indices go up to 2*n-2
    max_index = 2 * n - 2
    assert np.all(linkage[:, 0] < max_index)
    assert np.all(linkage[:, 1] < max_index)


# ---------------------------------------------------------------------------
# cut_tree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_clusters", [2, 3, 5])
def test_cut_tree_returns_correct_length(cluster_data, num_clusters):
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, LinkageMethod.Ward, DistanceMetric.Euclidean)
    labels = cut_tree(linkage, n, num_clusters)
    assert isinstance(labels, list)
    assert len(labels) == n


@pytest.mark.parametrize("num_clusters", [2, 3, 5])
def test_cut_tree_num_unique_labels(cluster_data, num_clusters):
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, LinkageMethod.Ward, DistanceMetric.Euclidean)
    labels = cut_tree(linkage, n, num_clusters)
    assert len(set(labels)) == num_clusters


def test_cut_tree_single_cluster(cluster_data):
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, LinkageMethod.Average, DistanceMetric.Euclidean)
    labels = cut_tree(linkage, n, 1)
    assert len(set(labels)) == 1


def test_cut_tree_all_singletons(cluster_data):
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, LinkageMethod.Average, DistanceMetric.Euclidean)
    labels = cut_tree(linkage, n, n)
    assert len(set(labels)) == n


# ---------------------------------------------------------------------------
# Input guards: the C++ core rejects non-finite data and
# malformed linkage matrices with ValueError instead of indexing out of range.
# ---------------------------------------------------------------------------

def test_non_finite_input_is_rejected(cluster_data):
    X, n, p = cluster_data
    bad = X.copy()
    bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        agglomerative_cluster(bad, LinkageMethod.Single, DistanceMetric.Euclidean)
    bad[1, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        agglomerative_cluster(bad, LinkageMethod.Average, DistanceMetric.Correlation)


def test_cut_tree_rejects_malformed_linkage(cluster_data):
    X, n, p = cluster_data
    linkage = agglomerative_cluster(X, LinkageMethod.Single, DistanceMetric.Euclidean)

    # Too few rows for the requested cut; the same prefix still cuts coarser.
    with pytest.raises(ValueError, match="rows"):
        cut_tree(linkage[:2], n, 1)
    assert len(cut_tree(linkage[:2], n, n - 2)) == n

    # A cluster id that does not exist yet at its step (0-based n at merge 0).
    bad = linkage.copy()
    bad[0, 1] = n
    with pytest.raises(ValueError, match="does not exist"):
        cut_tree(bad, n, 2)

    # A self-merge.
    bad = linkage.copy()
    bad[0, 1] = bad[0, 0]
    with pytest.raises(ValueError, match="itself"):
        cut_tree(bad, n, 2)

    # An id consumed twice.
    bad = linkage.copy()
    bad[1, 0] = linkage[0, 0]
    with pytest.raises(ValueError, match="already consumed"):
        cut_tree(bad, n, 2)


# ---------------------------------------------------------------------------
# Backend line
# ---------------------------------------------------------------------------

def test_verbose_default_is_silent(cluster_data, capfd):
    """The backend's allocation line prints only on request."""
    import ctypes
    X, n, p = cluster_data

    def c_stdout():
        # The line is written by the C++ backend; flush the C stdio buffer before reading
        ctypes.CDLL(None).fflush(None)
        return capfd.readouterr().out

    agglomerative_cluster(X, LinkageMethod.Ward, DistanceMetric.Euclidean)
    assert "allocated" not in c_stdout()
    agglomerative_cluster(X, LinkageMethod.Ward, DistanceMetric.Euclidean, verbose=True)
    assert "allocated" in c_stdout()
