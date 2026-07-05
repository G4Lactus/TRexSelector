"""
Terminating Solvers
"""
from __future__ import annotations
import typing
from . import afs_based
from . import lars_based
from . import omp_based
__all__: list[str] = ['L2', 'ScalingMode', 'ZSCORE', 'afs_based', 'lars_based', 'omp_based']
class ScalingMode:
    """
    Column scaling convention applied by the solver after centering.
    
    Members:
    
      L2 : Divide each centered column by its L2 norm (unit L2 norm).
    
      ZSCORE : Divide each centered column by its sample SD (n-1).
    """
    L2: typing.ClassVar[ScalingMode]  # value = <ScalingMode.L2: 0>
    ZSCORE: typing.ClassVar[ScalingMode]  # value = <ScalingMode.ZSCORE: 1>
    __members__: typing.ClassVar[dict[str, ScalingMode]]  # value = {'L2': <ScalingMode.L2: 0>, 'ZSCORE': <ScalingMode.ZSCORE: 1>}
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
L2: ScalingMode  # value = <ScalingMode.L2: 0>
ZSCORE: ScalingMode  # value = <ScalingMode.ZSCORE: 1>
