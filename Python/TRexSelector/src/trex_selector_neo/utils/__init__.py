# Utility classes from the C++ core
from .._core.utils import (
    AccessMode,
    MemoryMappedMatrix,
    convert_to_memory_mapped,
    compute_fdp,
    compute_tpp,
    compute_precision,
    compute_recall,
    compute_fdp_dense,
    compute_tpp_dense,
    true_positives_count,
    false_positives_count,
    false_negatives_count,
    f1_score,
    get_max_threads,
    set_num_threads,
)
import numpy as np


def numpy_to_memmap(filename: str, data: np.ndarray) -> MemoryMappedMatrix:
    """
    Convenience pipeline to port an existing dataset (e.g. NumPy array)
    into a MemoryMappedMatrix backed by a binary file.

    Parameters
    ----------
    filename : str
        The path of the binary file to create.
    data : np.ndarray
        A 2D array-like dataset to port into the memory-mapped matrix.
        Will automatically be parsed safely.

    Returns
    -------
    MemoryMappedMatrix
        The memory-mapped matrix object representing the data on disk.
    """
    data_np = np.asarray(data, dtype=np.float64)
    if data_np.ndim != 2:
        raise ValueError(f"Dataset must be a 2D array, but got {data_np.ndim}D array.")

    rows, cols = data_np.shape

    # Initialize the memory mapped file via the C++ extension
    mmap_mat = MemoryMappedMatrix(filename, rows, cols, AccessMode.ReadWrite)

    # Get the zero-copy view of the memory mapped binary
    mmap_arr = mmap_mat.to_numpy()

    # Copy dataset into mapped memory buffer natively (saves to disk instantly)
    mmap_arr[:, :] = data_np

    return mmap_mat

__all__ = [
    "AccessMode",
    "MemoryMappedMatrix",
    "numpy_to_memmap",
    "convert_to_memory_mapped",
    "compute_fdp",
    "compute_tpp",
    "compute_precision",
    "compute_recall",
    "compute_fdp_dense",
    "compute_tpp_dense",
    "true_positives_count",
    "false_positives_count",
    "false_negatives_count",
    "f1_score",
    "get_max_threads",
    "set_num_threads",
]
