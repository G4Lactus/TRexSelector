import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

from . import ml_methods
from . import tsolvers
from . import utils

# Import the compiled C++ extension inside the nested modules
from .trex_selector_methods import (
    LLoopStrategy, TRexControlParameter, SelectionResult, TRexSelector as PyTRexSelector,
    TRexDASelector as PyTRexDASelector, TRexDAControlParameter, DAMethod, DASelectionResult,
    TRexGVSSelector as PyTRexGVSSelector, TRexGVSControlParameter, GVSType, GVSSelectionResult,
    TRexScreeningSelector as PyTRexScreeningSelector, ScreenTRexControlParameter, ScreenTRexMethod, ScreenTRexSelectionResult,
    TRexBiobankScreeningSelector as PyBiobankScreeningSelector, BiobankScreenTRexControl, BiobankScreenTRexResult,
    SolverTypeForTRex, SolverHyperparameters, DummyDistribution,
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
    def K(self) -> int:
        return self._selector.getK()

    @property
    def num_selected(self) -> int:
        return self._selector.getNumSelected()

    @property
    def T_stop(self) -> int:
        return self._selector.getTStop()

    @property
    def L(self) -> int:
        return self._selector.getDummyMultiplierL()

    @property
    def v_thresh(self) -> float:
        return self._selector.getVotingThreshold()

    @property
    def voting_grid(self) -> np.ndarray:
        return self._selector.getVotingGrid()

    @property
    def fdp_hat_mat(self) -> np.ndarray:
        return self._selector.getFDPHatMat()

    @property
    def phi_mat(self) -> np.ndarray:
        return self._selector.getPhiMat()

    @property
    def r_mat(self) -> np.ndarray:
        return self._selector.getRMat()

    @property
    def phi_prime(self) -> np.ndarray:
        return self._selector.getPhiPrime()

    @property
    def x_means(self) -> np.ndarray:
        return self._selector.getXMeans()

    @property
    def x_l2_norms(self) -> np.ndarray:
        return self._selector.getXL2Norms()

    @property
    def y_mean(self) -> float:
        return self._selector.getYMean()

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

    def select(self) -> DASelectionResult:
        self._selector.select()
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

        # Derive solver_type from gvs_type — mirrors R package behaviour.
        # EN (Ordinary Elastic Net) requires TENET; IEN (Informed EN) requires TLASSO.
        trex_control.solver_type = (SolverTypeForTRex.TLASSO
                                    if gvs_control.gvs_type == GVSType.IEN
                                    else SolverTypeForTRex.TENET)

        self._selector = PyTRexGVSSelector(self.X, self.y, tFDR, gvs_control, trex_control, seed, verbose)

    def select(self) -> GVSSelectionResult:
        self._selector.select()
        return self._selector.getGVSResult()


class TRexScreeningSelector(TRexSelector):
    """
    Python wrapper for the C++ TRexScreeningSelector.
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
        # ScreenTRex only accepts STANDARD or PERMUTATION lloop_strategy.
        if trex_control.lloop_strategy not in (LLoopStrategy.STANDARD, LLoopStrategy.PERMUTATION):
            trex_control.lloop_strategy = LLoopStrategy.STANDARD

        self._selector = PyTRexScreeningSelector(self.X, self.y, screen_control, trex_control, seed, verbose)

    def select(self) -> ScreenTRexSelectionResult:
        self._selector.select()
        return self._selector.getScreenResult()


class TRexBiobankScreeningSelector:
    """
    Python wrapper for the C++ TRexBiobankScreeningSelector.

    Dispatches to screenPhenotype() for a single 1-D response vector
    or screenPhenotypes() for a 2-D response matrix.
    """

    def __init__(self, X: np.ndarray, Y,
                 biosctrex_ctrl: BiobankScreenTRexControl = None,
                 seed: int = -1, verbose: bool = False):
        self.X = np.asfortranarray(X, dtype=np.float64)
        if biosctrex_ctrl is None:
            biosctrex_ctrl = BiobankScreenTRexControl()
        Y = np.asarray(Y, dtype=np.float64)
        self._is_2d = Y.ndim > 1
        self._Y = np.asfortranarray(Y)
        self._selector = PyBiobankScreeningSelector(self.X, self._Y, biosctrex_ctrl, seed, verbose)

    def select(self):
        """
        Run Biobank screening.

        Returns
        -------
        BiobankScreenTRexResult
            For a 1-D response (single phenotype).
        list[BiobankScreenTRexResult]
            For a 2-D response (multiple phenotypes).
        """
        if self._is_2d:
            return self._selector.screenPhenotypes()
        return self._selector.screenPhenotype()


__all__ = [
    "LLoopStrategy",
    "SolverTypeForTRex",
    "SolverHyperparameters",
    "DummyDistribution",
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
    "TRexScreeningSelector",
    "ScreenTRexControlParameter",
    "ScreenTRexMethod",
    "ScreenTRexSelectionResult",
    "TRexBiobankScreeningSelector",
    "BiobankScreenTRexControl",
    "BiobankScreenTRexResult",
    # Submodules
    "ml_methods",
    "tsolvers",
    "utils",
]
