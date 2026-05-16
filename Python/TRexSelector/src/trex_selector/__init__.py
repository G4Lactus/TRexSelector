import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

# Import the compiled C++ extension inside the nested modules
from .trex_selector_methods import LLoopStrategy, TRexControlParameter, SelectionResult, TRexSelector as PyTRexSelector

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
