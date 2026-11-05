"""
Utility classes and functions
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['AccessMode', 'MemoryMappedMatrix', 'ReadOnly', 'ReadWrite', 'compute_fdp', 'compute_fdp_dense', 'compute_precision', 'compute_recall', 'compute_tpp', 'compute_tpp_dense', 'convert_to_memory_mapped', 'f1_score', 'false_negatives_count', 'false_positives_count', 'get_max_threads', 'get_num_threads', 'set_num_threads', 'true_positives_count']
class AccessMode:
    """
    Members:
    
      ReadOnly
    
      ReadWrite
    """
    ReadOnly: typing.ClassVar[AccessMode]  # value = <AccessMode.ReadOnly: 0>
    ReadWrite: typing.ClassVar[AccessMode]  # value = <AccessMode.ReadWrite: 1>
    __members__: typing.ClassVar[dict[str, AccessMode]]  # value = {'ReadOnly': <AccessMode.ReadOnly: 0>, 'ReadWrite': <AccessMode.ReadWrite: 1>}
    @typing.overload
    def __eq__(self, other: AccessMode) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: AccessMode) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MemoryMappedMatrix:
    """
    Column-major matrix of doubles backed by a memory-mapped file: elements and blocks are read
    and written in place on disk, and to_numpy() gives a zero-copy view for the out-of-core
    paths.
    
    References
    ----------
    F. Scheidt, J. Machkour, and M. Muma, "Solving FDR-Controlled Sparse Regression Problems
      with Five Million Variables on a Laptop," in Proc. IEEE 9th Int. Workshop Comput. Adv.
      Multi-Sensor Adapt. Process. (CAMSAP), 2023, pp. 116-120.
    """
    def __getitem__(self, idx: tuple[typing.SupportsInt | typing.SupportsIndex, typing.SupportsInt | typing.SupportsIndex]) -> float:
        """
        Read single element at (row, col) — 0-based indices.
        """
    def __init__(self, filename: str, rows: typing.SupportsInt | typing.SupportsIndex, cols: typing.SupportsInt | typing.SupportsIndex, mode: AccessMode = ...) -> None:
        ...
    @typing.overload
    def __setitem__(self, idx: tuple[typing.SupportsInt | typing.SupportsIndex, typing.SupportsInt | typing.SupportsIndex], value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Write single element at (row, col) — 0-based indices. Requires ReadWrite mode.
        """
    @typing.overload
    def __setitem__(self, idx: tuple, value: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Write a 2D NumPy array into the block selected by int/slice indices. Requires ReadWrite mode. Indices are 0-based.
        """
    def cols(self) -> int:
        ...
    def rows(self) -> int:
        ...
    def size(self) -> int:
        ...
    def to_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Returns a zero-copy NumPy array view of the memory-mapped matrix.
        """
    def write_block(self, row_start: typing.SupportsInt | typing.SupportsIndex, row_count: typing.SupportsInt | typing.SupportsIndex, col_start: typing.SupportsInt | typing.SupportsIndex, col_count: typing.SupportsInt | typing.SupportsIndex, values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        """
        Write a 2D NumPy block into the mmap at (row_start:row_start+row_count, col_start:col_start+col_count). Requires ReadWrite mode. Indices are 0-based.
        """
def compute_fdp(selected_indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], true_support: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> float:
    """
    Compute FDP from indices
    """
def compute_fdp_dense(beta_hat: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], beta: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], eps: typing.SupportsFloat | typing.SupportsIndex = 1e-15) -> float:
    """
    Compute FDP from dense coefficient vectors
    """
def compute_precision(selected_indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], true_support: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> float:
    """
    Compute precision from indices
    """
def compute_recall(selected_indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], true_support: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> float:
    """
    Compute recall from indices
    """
def compute_tpp(selected_indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], true_support: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> float:
    """
    Compute TPP from indices
    """
def compute_tpp_dense(beta_hat: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], beta: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], eps: typing.SupportsFloat | typing.SupportsIndex = 1e-15) -> float:
    """
    Compute TPP from dense coefficient vectors
    """
def convert_to_memory_mapped(array: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], filename: str) -> None:
    """
    Converts a NumPy array to a memory-mapped binary file.
    """
def f1_score(precision: typing.SupportsFloat | typing.SupportsIndex, recall: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute the F1 score from precision and recall
    """
def false_negatives_count(selected_indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], true_support: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> int:
    """
    Count false negatives (in true support but not selected)
    """
def false_positives_count(selected_indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], true_support: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> int:
    """
    Count false positives (selected but not in true support)
    """
def get_max_threads() -> int:
    """
    The threads OpenMP can run on this machine (1 in a build without OpenMP): an upper bound for a budget, not the budget itself.
    """
def get_num_threads() -> int:
    """
    The library's default thread budget, initially 1. The library is single-threaded by default: every parallel region runs on an explicit budget, a selector's random experiments and inner kernels on TRexControlParameter.max_outer_threads / max_inner_threads, a T-solver on SolverOptions.n_threads, and every method without an option of its own (the scalers, the clustering, the ML regressions, normalize_design) on this default budget, which their n_threads argument overrides per object.
    """
def set_num_threads(num_threads: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Set the library's default thread budget (>= 1) for the process and return the previous budget; see get_num_threads().
    """
def true_positives_count(selected_indices: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], true_support: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> int:
    """
    Count true positives (selected AND in true support)
    """
ReadOnly: AccessMode  # value = <AccessMode.ReadOnly: 0>
ReadWrite: AccessMode  # value = <AccessMode.ReadWrite: 1>
