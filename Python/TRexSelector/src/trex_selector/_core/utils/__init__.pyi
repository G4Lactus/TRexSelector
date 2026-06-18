from typing import overload

import numpy as np

# ---------------------------------------------------------------------------
# AccessMode
# ---------------------------------------------------------------------------

class AccessMode:
    ReadOnly: AccessMode
    ReadWrite: AccessMode

# ---------------------------------------------------------------------------
# MemoryMappedMatrix
# ---------------------------------------------------------------------------

class MemoryMappedMatrix:
    def __init__(
        self,
        filename: str,
        rows: int,
        cols: int,
        mode: AccessMode = ...,
    ) -> None: ...
    def rows(self) -> int: ...
    def cols(self) -> int: ...
    def size(self) -> int: ...
    def to_numpy(self) -> np.ndarray: ...
    def __getitem__(self, idx: tuple[int, int]) -> float: ...
    @overload
    def __setitem__(self, idx: tuple[int, int], value: float) -> None: ...
    @overload
    def __setitem__(self, idx: tuple[int | slice, int | slice], value: np.ndarray) -> None: ...
    def write_block(
        self,
        row_start: int,
        row_count: int,
        col_start: int,
        col_count: int,
        values: np.ndarray,
    ) -> None: ...

def convert_to_memory_mapped(array: np.ndarray, filename: str) -> None: ...

# ---------------------------------------------------------------------------
# Eval Metrics
# ---------------------------------------------------------------------------

def compute_fdp(selected_indices: list[int], true_support: list[int]) -> float: ...
def compute_tpp(selected_indices: list[int], true_support: list[int]) -> float: ...
def compute_precision(selected_indices: list[int], true_support: list[int]) -> float: ...
def compute_recall(selected_indices: list[int], true_support: list[int]) -> float: ...
def compute_fdp_dense(beta_hat: np.ndarray, beta: np.ndarray, eps: float = ...) -> float: ...
def compute_tpp_dense(beta_hat: np.ndarray, beta: np.ndarray, eps: float = ...) -> float: ...

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------

def get_max_threads() -> int: ...
def set_num_threads(num_threads: int) -> None: ...
