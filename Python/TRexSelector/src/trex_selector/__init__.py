import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

# Import the compiled C++ extension inside the nested modules
from .trex_selector_methods import (
    LLoopStrategy, TRexControlParameter, SelectionResult, TRexSelector as PyTRexSelector,
    TRexDASelector as PyTRexDASelector, TRexDAControlParameter, DAMethod, DASelectionResult,
    TRexGVSSelector as PyTRexGVSSelector, TRexGVSControlParameter, GVSType, GVSSelectionResult,
    ScreenTRexSelector as PyScreenTRexSelector, ScreenTRexControlParameter, ScreenTRexMethod, ScreenTRexSelectionResult
)

class TRexSelector:
    """
    Python wrapper for the C++ TRexSelector.

    This class performs FDR-controlled variable selection using the T-Rex algorithm.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, tFDR: float = 0.1,
                 trex_control: TRexControlParameter = None,
                 seed: int = -1, verbose: bool = True):
        # Validate data types and ensure Fortran contiguity for Eigen
        # Eigen Map defaults to ColumnMajor (Fortran order).
        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)

        if trex_control is None:
            trex_control = TRexControlParameter()

        self._selector = PyTRexSelector(self.X, self.y, tFDR, trex_control, seed, verbose)

    def select(self) -> SelectionResult:
        """
        Executes the T-Rex selector algorithm.
        Releases the GIL to allow underlying C++ OpenMP multithreading to run in parallel.

        Returns
        -------
        SelectionResult
            A data class containing the selected variables and extensive metadata.
        """
        return self._selector.select()

    @property
    def tFDR(self) -> float:
        return self._selector.getTFDR()

    @property
    def selected_var(self) -> np.ndarray:
        return self._selector.getSelectedVar()

    @property
    def selected_indices(self) -> list:
        return self._selector.getSelectedIndices()


class TRexDASelector(TRexSelector):
    """
    Python wrapper for the C++ TRexDASelector (Data Augmentation).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, tFDR: float = 0.1,
                 da_control: TRexDAControlParameter = None,
                 trex_control: TRexControlParameter = None,
                 seed: int = -1, verbose: bool = True):
        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)

        if da_control is None:
            da_control = TRexDAControlParameter()
        if trex_control is None:
            trex_control = TRexControlParameter()

        self._selector = PyTRexDASelector(self.X, self.y, tFDR, da_control, trex_control, seed, verbose)

    def select(self) -> SelectionResult:
        return self._selector.select()
        
    @property
    def da_result(self) -> DASelectionResult:
        return self._selector.getDAResult()


class TRexGVSSelector(TRexSelector):
    """
    Python wrapper for the C++ TRexGVSSelector (Group Variable Selection).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, tFDR: float = 0.1,
                 gvs_control: TRexGVSControlParameter = None,
                 trex_control: TRexControlParameter = None,
                 seed: int = -1, verbose: bool = True):
        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)

        if gvs_control is None:
            gvs_control = TRexGVSControlParameter()
        if trex_control is None:
            trex_control = TRexControlParameter()

        self._selector = PyTRexGVSSelector(self.X, self.y, tFDR, gvs_control, trex_control, seed, verbose)

    def select(self) -> SelectionResult:
        return self._selector.select()

    @property
    def gvs_result(self) -> GVSSelectionResult:
        return self._selector.getGVSResult()


class ScreenTRexSelector(TRexSelector):
    """
    Python wrapper for the C++ ScreenTRexSelector.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 screen_control: ScreenTRexControlParameter = None,
                 trex_control: TRexControlParameter = None,
                 seed: int = -1, verbose: bool = True):
        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)

        if screen_control is None:
            screen_control = ScreenTRexControlParameter()
        if trex_control is None:
            trex_control = TRexControlParameter()

        self._selector = PyScreenTRexSelector(self.X, self.y, screen_control, trex_control, seed, verbose)

    def select(self) -> SelectionResult:
        return self._selector.select()

    @property
    def screen_result(self) -> ScreenTRexSelectionResult:
        return self._selector.getScreenResult()

__all__ = [
    "LLoopStrategy",
    "TRexControlParameter",
    "SelectionResult",
    "TRexSelector",
    "TRexDASelector",
    "TRexDAControlParameter",
    "DAMethod",
    "DASelectionResult",
    "TRexGVSSelector",
    "TRexGVSControlParameter",
    "GVSType",
    "GVSSelectionResult",
    "ScreenTRexSelector",
    "ScreenTRexControlParameter",
    "ScreenTRexMethod",
    "ScreenTRexSelectionResult",
]
