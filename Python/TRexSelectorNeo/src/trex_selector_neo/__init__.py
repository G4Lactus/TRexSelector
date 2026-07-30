from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

from importlib.metadata import PackageNotFoundError, version as _pkg_version

import numpy as np

try:
    __version__ = _pkg_version("TRexSelectorNeo")
except PackageNotFoundError:  # Running from a source tree without install
    __version__ = "0+unknown"

from . import ml_methods
from . import tsolvers
from . import utils

# Import the compiled C++ extension inside the nested modules
from .trex_selector_methods import (
    LLoopStrategy, ScalingMode, TRexControlParameter, SelectionResult, TRexSelector as PyTRexSelector,
    TRexDASelector as PyTRexDASelector, TRexDAControlParameter, DAMethod, BTSelectionMode, DASelectionResult,
    TRexGVSSelector as PyTRexGVSSelector, TRexGVSControlParameter, GVSType, ENSolverType,
                        LambdaSelectionMethod, GVSSelectionResult,
    TRexScreeningSelector as PyTRexScreeningSelector, ScreenTRexControlParameter, ScreenTRexMethod,
                             ScreenTRexSelectionResult,
    TRexBiobankScreeningSelector as PyBiobankScreeningSelector, BiobankScreenTRexControl,
                                    BiobankScreenTRexResult,
    SolverTypeForTRex, SolverHyperparameters, DummyDistribution,
    SPCAMode, TRexSPCAControlParameter, TRexSPCAResult,
    TRexSPCASelector as PyTRexSPCASelector,
)


def _validate_core_solver_type(trex_control: "TRexControlParameter") -> None:
    """Reject solver types the core dispatch cannot serve, at construction.

    The shared C++ dispatch throws the same guidance at SELECT time, but from
    inside the OpenMP experiment loop, where the exception cannot cross the
    region boundary and terminates the process instead of raising. Used by
    the plain/DA/screening/biobank wrappers; the GVS wrapper deliberately
    bypasses it (the IEN family is exactly its vocabulary).
    """
    st = trex_control.solver_type
    if st == SolverTypeForTRex.TIENET:
        raise ValueError(
            'solver_type = TIENET is a GVS-only solver (needs a group '
            'assignment); use TRexGVSSelector with gvs_type = IEN.')
    if st == SolverTypeForTRex.TIENET_AUG:
        raise ValueError(
            'solver_type = TIENET_AUG is a GVS-only solver; use '
            'TRexGVSSelector with gvs_type = IEN.')
    if st == SolverTypeForTRex.TCIENET:
        raise ValueError(
            'solver_type = TCIENET needs a group assignment; use '
            'TRexGVSSelector with gvs_type = IEN.')


class TRexSelector:
    """
    Python wrapper for the C++ TRexSelector.

    This class performs FDR-controlled variable selection using the T-Rex selector.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, tFDR: float = 0.1,
                 trex_control: TRexControlParameter | None = None,
                 seed: int = -1, verbose: bool = True):

        # Validate data types and ensure Fortran contiguity for Eigen
        # Eigen Map defaults to ColumnMajor (Fortran order)
        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)

        if trex_control is None:
            trex_control = TRexControlParameter()
        _validate_core_solver_type(trex_control)

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
    def selected_indices(self) -> np.ndarray:
        return self._selector.getSelectedIndices()


class TRexDASelector(TRexSelector):
    """
    Python wrapper for the C++ TRexDASelector (Dependency Awareness).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, tFDR: float = 0.1,
                 da_control: TRexDAControlParameter | None = None,
                 trex_control: TRexControlParameter | None = None,
                 seed: int = -1, verbose: bool = True):

        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)

        if da_control is None:
            da_control = TRexDAControlParameter()
        if trex_control is None:
            trex_control = TRexControlParameter()
        _validate_core_solver_type(trex_control)

        self._selector: PyTRexDASelector = PyTRexDASelector(self.X, self.y, tFDR,
                                                            da_control,
                                                            trex_control,
                                                            seed, verbose)

    def select(self) -> DASelectionResult:
        self._selector.select()
        return self._selector.getDAResult()


class TRexGVSSelector(TRexSelector):
    """
    Python wrapper for the C++ TRexGVSSelector (Group Variable Selection).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, tFDR: float = 0.1,
                 gvs_control: TRexGVSControlParameter | None = None,
                 trex_control: TRexControlParameter | None = None,
                 seed: int = -1, verbose: bool = True):

        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)

        if gvs_control is None:
            gvs_control = TRexGVSControlParameter()
        if trex_control is None:
            trex_control = TRexControlParameter()

        # Solver selection mirrors the R surface:
        #   EN  -> derived from en_solver (TENET / TENET_AUG / TCENET);
        #          trex_control.solver_type is ignored (warn on an explicit
        #          non-matching value; TLARS is the default and cannot be
        #          distinguished from an explicit choice, so it never warns).
        #   IEN -> trex_control.solver_type IS the IEN-solver choice:
        #          TIENET (native pathwise, default), TIENET_AUG, TCIENET.
        #          The default TLARS maps to TIENET; any other value warns
        #          and falls back to TIENET (in the C++ glue).
        _ien_family = (SolverTypeForTRex.TIENET, SolverTypeForTRex.TIENET_AUG,
                       SolverTypeForTRex.TCIENET)
        if gvs_control.gvs_type == GVSType.IEN:
            if trex_control.solver_type not in (SolverTypeForTRex.TLARS,) + _ien_family:
                warnings.warn(
                    f"trex_control.solver_type = {trex_control.solver_type.name} "
                    f"is not an IEN-family solver; gvs_type = IEN accepts "
                    f"TIENET, TIENET_AUG, or TCIENET. Using TIENET.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            if gvs_control.en_solver == ENSolverType.TENET_AUG:
                derived_solver = SolverTypeForTRex.TENET_AUG
            elif gvs_control.en_solver == ENSolverType.TCENET:
                derived_solver = SolverTypeForTRex.TCENET
            else:
                derived_solver = SolverTypeForTRex.TENET
            if trex_control.solver_type not in (SolverTypeForTRex.TLARS, derived_solver):
                warnings.warn(
                    f"trex_control.solver_type = {trex_control.solver_type.name} is "
                    f"ignored by TRexGVSSelector: gvs_type = EN "
                    f"derives solver {derived_solver.name} from en_solver.",
                    UserWarning,
                    stacklevel=2,
                )

        # lambda2_method resolves by track: the struct default (CV_1SE_CCD,
        # indistinguishable from an explicit choice — same convention as
        # TLARS above) becomes the IEN-geometry tuner on the IEN track.
        # IEN users wanting an EN-shaped ridge CV explicitly still have
        # CV_MIN_CCD / CV_1SE_SVD / CV_MIN_SVD. The IEN-geometry tuners
        # consume the group structure, so the EN track rejects them.
        _ien_methods = (LambdaSelectionMethod.CV_1SE_IEN_CCD,
                        LambdaSelectionMethod.CV_MIN_IEN_CCD,
                        LambdaSelectionMethod.CV_1SE_TIK_SVD,
                        LambdaSelectionMethod.CV_MIN_TIK_SVD)
        _l2m_original = gvs_control.lambda2_method
        if gvs_control.gvs_type == GVSType.IEN:
            if gvs_control.lambda2_method == LambdaSelectionMethod.CV_1SE_CCD:
                gvs_control.lambda2_method = LambdaSelectionMethod.CV_1SE_IEN_CCD
        elif gvs_control.lambda2_method in _ien_methods:
            raise ValueError(
                f"lambda2_method = {gvs_control.lambda2_method.name} is an "
                f"IEN-geometry tuner (it consumes the group structure) and "
                f"requires gvs_type = IEN.")

        # lambda_2 == 0 is the degenerate no-ridge case (pure T-LASSO), NOT
        # automatic CV (the sentinel for that is lambda_2 < 0). Warn so a user
        # who passed exactly 0 expecting CV is not silently given no ridge.
        if gvs_control.lambda_2 == 0.0:
            warnings.warn(
                "gvs_control.lambda_2 = 0 is the degenerate no-ridge case "
                "(pure T-LASSO), not automatic cross-validation. Use "
                "lambda_2 < 0 (e.g. -1) to auto-determine lambda_2 via CV.",
                UserWarning,
                stacklevel=2,
            )
        try:
            self._selector: PyTRexGVSSelector = PyTRexGVSSelector(self.X, self.y, tFDR,
                                                                  gvs_control,
                                                                  trex_control,
                                                                  seed, verbose)
        finally:
            # The C++ selector copied the struct; undo the track resolution
            # so the caller's control object is left as they set it.
            gvs_control.lambda2_method = _l2m_original

    def select(self) -> GVSSelectionResult:
        self._selector.select()
        return self._selector.getGVSResult()


class TRexScreeningSelector(TRexSelector):
    """
    Python wrapper for the C++ TRexScreeningSelector.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 screen_control: ScreenTRexControlParameter | None = None,
                 trex_control: TRexControlParameter | None = None,
                 seed: int = -1, verbose: bool = True):

        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)

        if screen_control is None:
            screen_control = ScreenTRexControlParameter()
        if trex_control is None:
            trex_control = TRexControlParameter()

        # ScreenTRex accepts STANDARD, SEEDED, PERMUTATION, and
        # PERMUTATION_SEEDED (screening is a single pass at L = p; the L-loop
        # growth strategies HCONCAT / SKIPL have no meaning there). The
        # trex_control default (HCONCAT) silently maps to STANDARD — it is
        # indistinguishable from an explicit choice, same convention as
        # TLARS elsewhere; any other unsupported value warns.
        _screen_ok = (LLoopStrategy.STANDARD, LLoopStrategy.SEEDED,
                      LLoopStrategy.PERMUTATION, LLoopStrategy.PERMUTATION_SEEDED)
        if trex_control.lloop_strategy not in _screen_ok:
            if trex_control.lloop_strategy != LLoopStrategy.HCONCAT:
                warnings.warn(
                    f"lloop_strategy = {trex_control.lloop_strategy.name} is not "
                    f"supported by screening (single pass at L = p); using "
                    f"STANDARD.",
                    UserWarning,
                    stacklevel=2,
                )
            trex_control.lloop_strategy = LLoopStrategy.STANDARD
        _validate_core_solver_type(trex_control)

        self._selector: PyTRexScreeningSelector = PyTRexScreeningSelector(self.X, self.y,
                                                                          screen_control,
                                                                          trex_control,
                                                                          seed, verbose)

    def select(self) -> ScreenTRexSelectionResult:
        self._selector.select()
        return self._selector.getScreenResult()


class TRexBiobankScreeningSelector:
    """
    Python wrapper for the C++ TRexBiobankScreeningSelector orchestrator.

    Dispatches to screenPhenotype() for a single 1-D response vector
    or screenPhenotypes() for a m-D response matrix.
    """

    def __init__(self, X: np.ndarray, Y,
                 bio_ctrl: BiobankScreenTRexControl | None = None,
                 seed: int = -1, verbose: bool = False):

        self.X = np.asfortranarray(X, dtype=np.float64)
        if bio_ctrl is None:
            bio_ctrl = BiobankScreenTRexControl()

        _validate_core_solver_type(bio_ctrl.trex_screen_ctrl.trex_ctrl)

        Y = np.asarray(Y, dtype=np.float64)
        self._is_mD = Y.ndim > 1
        self._Y = np.asfortranarray(Y)
        self._selector = PyBiobankScreeningSelector(self.X, self._Y,
                                                    bio_ctrl,
                                                    seed, verbose)

    def select(self) -> BiobankScreenTRexResult | list[BiobankScreenTRexResult]:
        """
        Run Biobank screening.

        Returns
        -------
        BiobankScreenTRexResult
            For a 1-D response (single phenotype).
        list[BiobankScreenTRexResult]
            For a m-D response (multiple phenotypes).
        """
        if self._is_mD:
            return self._selector.screenPhenotypes()
        return self._selector.screenPhenotype()


class TRexSPCASelector:
    """
    Python wrapper for T-Rex Sparse PCA selection.

    Extracts sparse principal components under FDR control. For each of the
    M components, ordinary PCA identifies a PC-response direction, T-Rex GVS
    selects a sparse active set at the target FDR, and a sparse loading vector
    is assembled. Explained variance is adjusted for non-orthogonality via QR.
    """

    def __init__(self,
                 X: np.ndarray,
                 spca_ctrl: TRexSPCAControlParameter | None = None,
                 seed: int = -1):
        """
        Parameters
        ----------
        X : ndarray, shape (n, p)
            Design matrix. Stored by reference; centered in-place during
            ``select()`` and restored on return.
        spca_ctrl : TRexSPCAControlParameter, optional
            Algorithmic control. Defaults to ``TRexSPCAControlParameter()``.
            The GVS sub-selector always runs the EN variant
            (``gvs_ctrl.gvs_type`` is overridden internally; IEN is not used
            by T-Rex SPCA) — the EN solver flavour is chosen via
            ``spca_ctrl.en_solver``. The GVS ridge penalty follows
            ``gvs_ctrl.lambda_2``: ``< 0`` (default) auto-determines it via
            ``gvs_ctrl.lambda2_method``, ``0`` is the degenerate pure T-LASSO
            case, and ``> 0`` fixes it in LARS units.
        seed : int, optional
            Random seed. ``-1`` (default) for non-deterministic (hardware
            entropy). ``>= 0`` for reproducible per-PC runs.
        """
        self.X = np.asfortranarray(X, dtype=np.float64)
        self._ctrl = spca_ctrl if spca_ctrl is not None else TRexSPCAControlParameter()
        self._seed = seed
        self._result: TRexSPCAResult | None = None

        # lambda_2 == 0 is the degenerate no-ridge case (pure T-LASSO), NOT
        # automatic CV (the sentinel for that is lambda_2 < 0). Warn so a user
        # who set exactly 0 expecting CV is not silently given no ridge.
        if self._ctrl.gvs_ctrl.lambda_2 == 0.0:
            warnings.warn(
                "spca_ctrl.gvs_ctrl.lambda_2 = 0 is the degenerate no-ridge "
                "case (pure T-LASSO), not automatic cross-validation. Use "
                "lambda_2 < 0 (e.g. -1) to auto-determine lambda_2 via CV.",
                UserWarning,
                stacklevel=2,
            )

    def select(self, M: int, tFDR: float) -> "TRexSPCASelector":
        """
        Run T-Rex Sparse PCA selection.

        Parameters
        ----------
        M : int
            Number of sparse principal components to extract.
        tFDR : float
            Target false discovery rate in (0, 1).

        Returns
        -------
        self : TRexSPCASelector
            For method chaining. Results accessible via properties.
        """
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        if not 0.0 < tFDR < 1.0:
            raise ValueError(f"tFDR must be in (0, 1), got {tFDR}")
        selector = PyTRexSPCASelector(self.X, M, tFDR, self._ctrl, self._seed, False)
        selector.select()
        self._result = selector.getResult()
        return self

    @property
    def Z(self) -> np.ndarray:
        """Score matrix (n × M) from the last ``select()`` call."""
        if self._result is None:
            raise RuntimeError("Call select() first.")
        return self._result.Z

    @property
    def V(self) -> np.ndarray:
        """Loading matrix (p × M) from the last ``select()`` call."""
        if self._result is None:
            raise RuntimeError("Call select() first.")
        return self._result.V

    @property
    def active_sets(self) -> list[list[int]]:
        """Per-component active sets (0-based indices) from the last ``select()`` call."""
        if self._result is None:
            raise RuntimeError("Call select() first.")
        return self._result.active_sets

    @property
    def adjusted_ev(self) -> np.ndarray:
        """Marginal adjusted explained variance per component (M-vector)."""
        if self._result is None:
            raise RuntimeError("Call select() first.")
        return self._result.adjusted_ev

    @property
    def cumulative_ev(self) -> np.ndarray:
        """Cumulative percentage of explained variance (M-vector)."""
        if self._result is None:
            raise RuntimeError("Call select() first.")
        return self._result.cumulative_ev


__all__ = [
    "LLoopStrategy",
    "ScalingMode",
    "SolverTypeForTRex",
    "SolverHyperparameters",
    "DummyDistribution",
    "TRexControlParameter",
    "SelectionResult",
    "TRexSelector",
    "TRexDASelector",
    "TRexDAControlParameter",
    "DAMethod",
    "BTSelectionMode",
    "DASelectionResult",
    "TRexGVSSelector",
    "TRexGVSControlParameter",
    "GVSType",
    "ENSolverType",
    "LambdaSelectionMethod",
    "GVSSelectionResult",
    "TRexScreeningSelector",
    "ScreenTRexControlParameter",
    "ScreenTRexMethod",
    "ScreenTRexSelectionResult",
    "TRexBiobankScreeningSelector",
    "BiobankScreenTRexControl",
    "BiobankScreenTRexResult",
    "SPCAMode",
    "TRexSPCAControlParameter",
    "TRexSPCAResult",
    "TRexSPCASelector",
    # Submodules
    "ml_methods",
    "tsolvers",
    "utils",
]
