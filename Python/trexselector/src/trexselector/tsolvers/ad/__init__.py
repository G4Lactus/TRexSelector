"""Augmented-dummy (AD) T-solvers: the classical family running on ``[X, D]``.

``SolverOptions``, ``ScalingMode`` and ``GroupPenaltyVariant`` live here rather
than one level up because every user of them is an AD solver (the same
placement rule as ``cpp/src/tsolvers/ad``).
"""
from ..._core.tsolvers.ad import ScalingMode, SolverOptions, GroupPenaltyVariant
from . import lars_based
from . import omp_based
from . import afs_based
from . import cd_based

__all__ = [
    "ScalingMode",
    "SolverOptions",
    "GroupPenaltyVariant",
    "lars_based",
    "omp_based",
    "afs_based",
    "cd_based",
]
