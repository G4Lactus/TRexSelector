"""Terminating solvers, one subpackage per dummy mechanism.

Mirrors ``cpp/src/tsolvers/{ad, vd, sd}``: ``ad`` holds the augmented-dummy
(classical) family that runs on the augmented design ``[X, D]``. The
virtual-dummy (``vd``) and sparse-dummy (``sd``) families join once they are
validated and bound. The families share no code and no class serves more than
one mechanism.
"""
from . import ad

__all__ = ["ad"]
