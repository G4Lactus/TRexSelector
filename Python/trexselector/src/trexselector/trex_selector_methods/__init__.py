"""T-Rex selectors, one subpackage per dummy mechanism.

Mirrors ``cpp/src/trex_selector_methods/{ad, vd, sd}``: ``ad`` holds the
augmented-dummy selectors (classical, DA, GVS, screening, SPCA, Tikhonov),
which the package root re-exports as the everyday spelling. The virtual-dummy
(``vd``) and sparse-dummy (``sd``) selectors join once they are validated and
bound, reachable only through their subpackage.
"""
from . import ad

__all__ = ["ad"]
