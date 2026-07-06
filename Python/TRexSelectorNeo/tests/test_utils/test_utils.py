"""
Tests for utility functions: MemoryMappedMatrix, evaluation metrics, OpenMP helpers.
"""
import numpy as np
import pytest
from trex_selector_neo.utils import (
    AccessMode,
    MemoryMappedMatrix,
    numpy_to_memmap,
    compute_fdp,
    compute_tpp,
    compute_precision,
    compute_recall,
    compute_fdp_dense,
    compute_tpp_dense,
    get_max_threads,
    set_num_threads,
)


# ---------------------------------------------------------------------------
# MemoryMappedMatrix round-trip
# ---------------------------------------------------------------------------

def test_memmap_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((15, 4)).astype(np.float64)
    filename = str(tmp_path / "test_mat.bin")
    mmap = numpy_to_memmap(filename, X)
    assert mmap.rows() == 15
    assert mmap.cols() == 4
    assert mmap.size() == 15 * 4
    assert np.allclose(mmap.to_numpy(), X)


def test_memmap_to_numpy_shape(tmp_path):
    X = np.eye(5, dtype=np.float64)
    mmap = numpy_to_memmap(str(tmp_path / "eye.bin"), X)
    arr = mmap.to_numpy()
    assert arr.shape == (5, 5)


def test_memmap_to_numpy_values(tmp_path):
    X = np.arange(12, dtype=np.float64).reshape(3, 4)
    mmap = numpy_to_memmap(str(tmp_path / "arange.bin"), X)
    assert np.allclose(mmap.to_numpy(), X)


def test_memmap_readonly_mode(tmp_path):
    X = np.ones((4, 3), dtype=np.float64)
    filename = str(tmp_path / "readonly.bin")
    numpy_to_memmap(filename, X)  # write first
    mmap_ro = MemoryMappedMatrix(filename, 4, 3, AccessMode.ReadOnly)
    assert mmap_ro.rows() == 4
    assert mmap_ro.cols() == 3


def test_numpy_to_memmap_rejects_1d(tmp_path):
    with pytest.raises(Exception):
        numpy_to_memmap(str(tmp_path / "bad.bin"), np.array([1.0, 2.0]))


def test_numpy_to_memmap_rejects_3d(tmp_path):
    with pytest.raises(Exception):
        numpy_to_memmap(str(tmp_path / "bad3d.bin"), np.ones((2, 3, 4)))


# ---------------------------------------------------------------------------
# MemoryMappedMatrix element access: __getitem__ / __setitem__
# ---------------------------------------------------------------------------

def test_memmap_getitem_returns_correct_value(tmp_path):
    X = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    mmap = numpy_to_memmap(str(tmp_path / "getitem.bin"), X)
    assert mmap[0, 0] == pytest.approx(X[0, 0])
    assert mmap[1, 2] == pytest.approx(X[1, 2])
    assert mmap[2, 3] == pytest.approx(X[2, 3])


def test_memmap_setitem_modifies_element(tmp_path):
    X = np.zeros((4, 3), dtype=np.float64)
    mmap = numpy_to_memmap(str(tmp_path / "setitem.bin"), X)
    mmap[1, 2] = 99.0
    assert mmap.to_numpy()[1, 2] == pytest.approx(99.0)
    # Neighbours untouched
    assert mmap.to_numpy()[0, 2] == pytest.approx(0.0)
    assert mmap.to_numpy()[1, 1] == pytest.approx(0.0)


def test_memmap_setitem_readonly_raises(tmp_path):
    X = np.ones((3, 3), dtype=np.float64)
    filename = str(tmp_path / "ro_write.bin")
    numpy_to_memmap(filename, X)
    mmap_ro = MemoryMappedMatrix(filename, 3, 3, AccessMode.ReadOnly)
    with pytest.raises(Exception):
        mmap_ro[0, 0] = 5.0


def test_memmap_getitem_out_of_bounds(tmp_path):
    X = np.zeros((3, 3), dtype=np.float64)
    mmap = numpy_to_memmap(str(tmp_path / "bounds.bin"), X)
    with pytest.raises(Exception):
        _ = mmap[3, 0]  # row == rows (0-based)
    with pytest.raises(Exception):
        _ = mmap[0, 3]  # col == cols (0-based)


# ---------------------------------------------------------------------------
# MemoryMappedMatrix block write: __setitem__ (slice) and write_block()
# ---------------------------------------------------------------------------

def test_memmap_setitem_block_roundtrip(tmp_path):
    X = np.zeros((6, 8), dtype=np.float64)
    mmap = numpy_to_memmap(str(tmp_path / "block.bin"), X)
    block = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    mmap[0:3, 1:5] = block
    result = mmap.to_numpy()[0:3, 1:5]
    assert np.allclose(result, block)
    # Neighbouring region must be untouched
    assert np.allclose(mmap.to_numpy()[3:, :], 0.0)
    assert np.allclose(mmap.to_numpy()[:, 0:1], 0.0)


def test_memmap_setitem_mixed_int_slice(tmp_path):
    X = np.zeros((5, 6), dtype=np.float64)
    mmap = numpy_to_memmap(str(tmp_path / "mixed.bin"), X)
    # Single row written as 1-row block (int index treated as degenerate slice)
    row_data = np.arange(6.0, dtype=np.float64).reshape(1, 6)
    mmap[2:3, 0:6] = row_data
    assert np.allclose(mmap.to_numpy()[2:3, :], row_data)
    # Single column written as 1-col block
    col_data = np.arange(5.0, dtype=np.float64).reshape(5, 1)
    mmap[0:5, 3:4] = col_data
    assert np.allclose(mmap.to_numpy()[:, 3:4], col_data)


def test_memmap_setitem_block_readonly_raises(tmp_path):
    X = np.ones((4, 4), dtype=np.float64)
    filename = str(tmp_path / "ro_block.bin")
    numpy_to_memmap(filename, X)
    mmap_ro = MemoryMappedMatrix(filename, 4, 4, AccessMode.ReadOnly)
    with pytest.raises(RuntimeError):
        mmap_ro[0:2, 0:2] = np.zeros((2, 2), dtype=np.float64)


def test_memmap_setitem_block_shape_mismatch(tmp_path):
    X = np.zeros((5, 5), dtype=np.float64)
    mmap = numpy_to_memmap(str(tmp_path / "mismatch.bin"), X)
    with pytest.raises(ValueError):
        mmap[0:3, 0:3] = np.zeros((2, 3), dtype=np.float64)  # wrong row count
    with pytest.raises(ValueError):
        mmap[0:3, 0:3] = np.zeros((3, 2), dtype=np.float64)  # wrong col count


def test_memmap_setitem_block_out_of_bounds(tmp_path):
    X = np.zeros((4, 4), dtype=np.float64)
    mmap = numpy_to_memmap(str(tmp_path / "oob_block.bin"), X)
    with pytest.raises(Exception):
        mmap[2:5, 0:2] = np.zeros((3, 2), dtype=np.float64)  # row slice exceeds dims
    with pytest.raises(Exception):
        mmap[0:2, 3:6] = np.zeros((2, 3), dtype=np.float64)  # col slice exceeds dims


def test_memmap_write_block_method_roundtrip(tmp_path):
    X = np.zeros((6, 8), dtype=np.float64)
    mmap = numpy_to_memmap(str(tmp_path / "wblock.bin"), X)
    block = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    mmap.write_block(0, 3, 1, 4, block)
    result = mmap.to_numpy()[0:3, 1:5]
    assert np.allclose(result, block)
    # Neighbouring region must be untouched
    assert np.allclose(mmap.to_numpy()[3:, :], 0.0)
    assert np.allclose(mmap.to_numpy()[:, 0:1], 0.0)


# ---------------------------------------------------------------------------
# compute_fdp
# ---------------------------------------------------------------------------

def test_fdp_empty_selection():
    assert compute_fdp([], [0, 1]) == pytest.approx(0.0)


def test_fdp_all_correct():
    assert compute_fdp([0, 1], [0, 1]) == pytest.approx(0.0)


def test_fdp_one_false_positive():
    # selected={0,2}, true={0,1} → FP={2}, FDP = 1/2
    assert compute_fdp([0, 2], [0, 1]) == pytest.approx(0.5)


def test_fdp_all_false_positives():
    # selected={2,3}, true={0,1} → FP={2,3}, FDP = 2/2 = 1
    assert compute_fdp([2, 3], [0, 1]) == pytest.approx(1.0)


def test_fdp_empty_true_support():
    # Null model: no true actives → every selection is a false positive
    assert compute_fdp([2, 3], []) == pytest.approx(1.0)
    # No selections under the null model → FDP defined as 0
    assert compute_fdp([], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_tpp
# ---------------------------------------------------------------------------

def test_tpp_empty_selection():
    assert compute_tpp([], [0, 1]) == pytest.approx(0.0)


def test_tpp_all_detected():
    assert compute_tpp([0, 1], [0, 1]) == pytest.approx(1.0)


def test_tpp_partial_detection():
    # selected={0,2}, true={0,1} → TP={0}, TPP = 1/2
    assert compute_tpp([0, 2], [0, 1]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_precision and compute_recall
# ---------------------------------------------------------------------------

def test_precision_perfect():
    assert compute_precision([0, 1], [0, 1]) == pytest.approx(1.0)


def test_precision_empty_selection():
    # No selections → precision is 0 (or defined as 1 by convention)
    result = compute_precision([], [0, 1])
    assert isinstance(result, float)


def test_recall_perfect():
    assert compute_recall([0, 1], [0, 1]) == pytest.approx(1.0)


def test_recall_empty_selection():
    assert compute_recall([], [0, 1]) == pytest.approx(0.0)


def test_recall_partial():
    # selected={0}, true={0,1} → recall = 1/2
    assert compute_recall([0], [0, 1]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_fdp_dense and compute_tpp_dense
# ---------------------------------------------------------------------------

def test_fdp_dense_one_false_positive():
    beta = np.array([3.0, -2.0, 0.0, 0.0, 0.0])
    beta_hat = np.array([2.5, 0.0, 1.0, 0.0, 0.0])
    # selected: {0, 2}, FP: {2}, FDP = 1/2
    assert compute_fdp_dense(beta_hat, beta) == pytest.approx(0.5)


def test_tpp_dense_partial():
    beta = np.array([3.0, -2.0, 0.0, 0.0, 0.0])
    beta_hat = np.array([2.5, 0.0, 1.0, 0.0, 0.0])
    # TP: {0}, true nonzero: {0,1}, TPP = 1/2
    assert compute_tpp_dense(beta_hat, beta) == pytest.approx(0.5)


def test_fdp_dense_all_correct():
    beta = np.array([1.0, -1.0, 0.0])
    beta_hat = np.array([0.8, -0.9, 0.0])
    assert compute_fdp_dense(beta_hat, beta) == pytest.approx(0.0)


def test_tpp_dense_all_recovered():
    beta = np.array([1.0, -1.0, 0.0])
    beta_hat = np.array([0.8, -0.9, 0.0])
    assert compute_tpp_dense(beta_hat, beta) == pytest.approx(1.0)


def test_fdp_dense_no_selection():
    beta = np.array([1.0, -1.0, 0.0])
    beta_hat = np.zeros(3)
    assert compute_fdp_dense(beta_hat, beta) == pytest.approx(0.0)


def test_tpp_dense_no_selection():
    beta = np.array([1.0, -1.0, 0.0])
    beta_hat = np.zeros(3)
    assert compute_tpp_dense(beta_hat, beta) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# OpenMP utilities
# ---------------------------------------------------------------------------

def test_get_max_threads_positive():
    t = get_max_threads()
    assert isinstance(t, int)
    assert t >= 1


def test_set_num_threads_no_exception():
    set_num_threads(1)  # should not raise


def test_set_num_threads_restore():
    original = get_max_threads()
    set_num_threads(1)
    set_num_threads(original)
