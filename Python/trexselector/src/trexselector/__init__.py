from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

from importlib.metadata import PackageNotFoundError, version as _pkg_version

import numpy as np

try:
    __version__ = _pkg_version("trexselector")
except PackageNotFoundError:  # Running from a source tree without install
    __version__ = "0+unknown"

from . import ml_methods
from . import tsolvers
from . import utils

# Import the compiled C++ extension inside the nested modules. The root
# re-exports the augmented-dummy (ad) selectors as the everyday spelling;
# the vd / sd families, once bound, stay under trex_selector_methods.vd / .sd.
from . import trex_selector_methods
from .trex_selector_methods.ad import (
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
    Lambda2CVDiagnostics,
    TikLambda2Method, PilotStatistic, TikPenaltyStructure,
    Lambda2PickInfo, Lambda2PilotResult, Lambda2Probe, Lambda2PowerRuleResult,
    TRexTikhonovControlParameter,
    TRexTikhonovSelector as PyTRexTikhonovSelector,
    gamma_to_k, build_group_centering_k,
    tikhonov_lambda2_power_rule as _tikhonov_lambda2_power_rule,
    normalize_design,
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
    if st == SolverTypeForTRex.TCDIENET:
        raise ValueError(
            'solver_type = TCDIENET needs a group assignment; use '
            'TRexGVSSelector with gvs_type = IEN.')
    if st in (SolverTypeForTRex.TCDTIKNET, SolverTypeForTRex.TIKNET):
        raise ValueError(
            f'solver_type = {st.name} needs a Tikhonov penalty matrix; use '
            'TRexTikhonovSelector.')


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

    def restore(self) -> None:
        """
        Give X back before select(): undo the in-place normalization applied at
        construction and release the buffer claim.

        Idempotent, and a no-op after select() (which restores X itself). A
        restored selector cannot run: select() then raises RuntimeError,
        construct a new one. Without it, an unselected selector gives X back
        only when it is garbage-collected.
        """
        self._selector.restore()

    @property
    def tFDR(self) -> float:
        return self._selector.getTFDR()

    @property
    def K(self) -> int:
        return self._selector.getK()

    @property
    def num_dummies(self) -> int:
        """Number of dummies used by the last select() (the L-loop's calibration)."""
        return self._selector.getNumDummies()

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
        #   EN  -> derived from en_solver (TENET / TENET_AUG / TCDENET);
        #          trex_control.solver_type is ignored (warn on an explicit
        #          non-matching value; TLARS is the default and cannot be
        #          distinguished from an explicit choice, so it never warns).
        #   IEN -> trex_control.solver_type IS the IEN-solver choice:
        #          TIENET (native pathwise, default), TIENET_AUG, TCDIENET.
        #          The default TLARS maps to TIENET; any other value warns
        #          and falls back to TIENET (in the C++ glue).
        _ien_family = (SolverTypeForTRex.TIENET, SolverTypeForTRex.TIENET_AUG,
                       SolverTypeForTRex.TCDIENET)
        if gvs_control.gvs_type == GVSType.IEN:
            if trex_control.solver_type not in (SolverTypeForTRex.TLARS,) + _ien_family:
                warnings.warn(
                    f"trex_control.solver_type = {trex_control.solver_type.name} "
                    f"is not an IEN-family solver; gvs_type = IEN accepts "
                    f"TIENET, TIENET_AUG, or TCDIENET. Using TIENET.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            if gvs_control.en_solver == ENSolverType.TENET_AUG:
                derived_solver = SolverTypeForTRex.TENET_AUG
            elif gvs_control.en_solver == ENSolverType.TCDENET:
                derived_solver = SolverTypeForTRex.TCDENET
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

        # lambda2_method resolves by track in the C++ constructor: the struct
        # default CV_1SE_CCD becomes CV_1SE_IEN_CCD on the IEN track, and the
        # IEN-geometry tuners are refused on the EN track (ValueError).

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
        self._selector: PyTRexGVSSelector = PyTRexGVSSelector(self.X, self.y, tFDR,
                                                              gvs_control,
                                                              trex_control,
                                                              seed, verbose)

    def select(self) -> GVSSelectionResult:
        self._selector.select()
        return self._selector.getGVSResult()

    def getLambda2CVDiagnostics(self):
        """Health record of the lambda_2 tuner of the last select().

        Returns a Lambda2CVDiagnostics: ``ran`` (a tuner ran; False when
        lambda_2 came from the caller), ``seconds``, ``n_profile_at_edge``
        (lambda_2 values whose profiled lambda_1 optimum sat on a
        truncation edge; 0 is healthy), ``support_cap`` (-1 = disabled),
        ``dfmax_ratio`` (IEN tuner: the ``cv_dfmax_ratio`` it was called
        with; 0 when no IEN-geometry tuner ran), ``low_dimensional``, and
        the boundary-pick guard fields ``grid_index`` / ``grid_size``
        (0-based position of the returned lambda_2 on the tuner's grid),
        ``at_grid_edge`` (first or last grid point: the grid boundary, not a
        CV optimum), ``profile_at_edge`` (IEN tuner: the chosen lambda_2's
        profiled lambda_1 optimum sat on the truncation edge),
        ``profile_at_support_cap`` (that edge is the support cap; raise
        ``cv_dfmax_ratio`` to move it) and ``boundary_pick`` (at_grid_edge
        or profile_at_edge; the selector warned once and returned the pick
        unchanged). The same record is carried in
        ``GVSSelectionResult.lambda2_cv``.
        """
        return self._selector.getLambda2CVDiagnostics()


class TRexTikhonovSelector(TRexSelector):
    """
    Python wrapper for the C++ TRexTikhonovSelector (general-Tikhonov T-Rex).

    The penalty comes from ``tik_control``: a dense p x p PSD matrix
    ``tikhonov_K`` (``penalty_structure = SPARSE_GENERAL``, converted to sparse
    internally) or ``GROUP_CENTERING`` built from the groups. lambda_2 < 0
    resolves by ``tik_control.lambda2_method`` (CV tuners or the pilot power
    rules); the dummies are cluster-aware by default. Solver: TCDTIKNET
    (default) or TIKNET, chosen through ``trex_control.solver_type``.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, tFDR: float = 0.1,
                 tik_control: TRexTikhonovControlParameter | None = None,
                 trex_control: TRexControlParameter | None = None,
                 seed: int = -1, verbose: bool = True):

        self.X = np.asfortranarray(X, dtype=np.float64)
        self.y = np.asfortranarray(y, dtype=np.float64)

        if tik_control is None:
            raise ValueError(
                "tik_control is required (a TRexTikhonovControlParameter; it "
                "carries the penalty matrix tikhonov_K or the penalty structure).")
        if trex_control is None:
            trex_control = TRexControlParameter()

        # Only the Tikhonov family has a general-K mode. TLARS is the
        # TRexControlParameter default and cannot be distinguished from an
        # explicit choice, so it never warns; the C++ glue maps it to
        # TCDTIKNET. Mirrors the R surface.
        _tik_family = (SolverTypeForTRex.TCDTIKNET, SolverTypeForTRex.TIKNET)
        if trex_control.solver_type not in (SolverTypeForTRex.TLARS,) + _tik_family:
            warnings.warn(
                f"trex_control.solver_type = {trex_control.solver_type.name} "
                f"is not a Tikhonov-family solver; TRexTikhonovSelector accepts "
                f"TCDTIKNET or TIKNET. Using TCDTIKNET.",
                UserWarning,
                stacklevel=2,
            )

        self._selector: PyTRexTikhonovSelector = PyTRexTikhonovSelector(
            self.X, self.y, tFDR, tik_control, trex_control, seed, verbose)

    @property
    def lambda2_used(self) -> float:
        """lambda_2 used by the last select() (solver scale); -1 before select()."""
        return self._selector.getLambda2Used()

    @property
    def lambda2_cv_diagnostics(self) -> Lambda2CVDiagnostics:
        """Health record of the lambda_2 tuner of the last select() (Lambda2CVDiagnostics)."""
        return self._selector.getLambda2CVDiagnostics()

    @property
    def lambda2_pilot(self) -> Lambda2PilotResult:
        """Record of the lambda_2 pilot of the last select() (Lambda2PilotResult; ``ran`` is False unless the
        pilot rule resolved lambda_2)."""
        return self._selector.getLambda2PilotResult()


def tikhonov_lambda2_power_rule(X: np.ndarray, y: np.ndarray, tFDR: float = 0.1,
                                tik_control: TRexTikhonovControlParameter | None = None,
                                trex_control: TRexControlParameter | None = None,
                                seed: int = 0,
                                grid: "list[float] | np.ndarray" = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
                                                                   1000.0, 3000.0),
                                knee_tolerance: float = 1.0,
                                verbose: bool = False) -> Lambda2PowerRuleResult:
    """
    The knee of the full lambda_2 profile of the Tikhonov T-Rex.

    Runs one FDR-controlled Tikhonov T-Rex selection per value of ``grid`` with
    the same seed, so the probes share their dummies and differ by the penalty
    weight alone, and picks the smallest lambda_2 whose selection is within
    ``knee_tolerance`` selections of the largest (the knee of the profile; 0
    reads the largest selection, ties toward the smallest lambda_2). This is
    the reference rule behind the default ``TikLambda2Method.PILOT_KNEE``,
    which reads the same knee off a few random experiments per grid value at a
    fraction of the cost; the full profile costs one selection per grid value
    and returns every probe's selection (``Lambda2PowerRuleResult``). The pick
    depends on y and lies outside the fixed-lambda_2 FDR guarantee that each
    probe carries.

    ``X`` and ``y`` are converted to Fortran-ordered float64 arrays once (no
    copy when they already are) and accessed zero-copy: every probe normalizes
    X in place and hands it back, so X is unchanged on return. ``tik_control``
    is required; its ``lambda_2`` is overwritten per probe. A negative seed
    would draw fresh dummies per probe and is refused.
    """
    if tik_control is None:
        raise ValueError(
            "tik_control is required (a TRexTikhonovControlParameter; it "
            "carries the penalty matrix tikhonov_K or the penalty structure).")
    if trex_control is None:
        trex_control = TRexControlParameter()
    if seed < 0:
        raise ValueError("seed must be non-negative: the probes share their dummies.")
    _tik_family = (SolverTypeForTRex.TCDTIKNET, SolverTypeForTRex.TIKNET)
    if trex_control.solver_type not in (SolverTypeForTRex.TLARS,) + _tik_family:
        warnings.warn(
            f"trex_control.solver_type = {trex_control.solver_type.name} "
            f"is not a Tikhonov-family solver; the Tikhonov T-Rex accepts "
            f"TCDTIKNET or TIKNET. Using TCDTIKNET.",
            UserWarning,
            stacklevel=2,
        )
    X = np.asfortranarray(X, dtype=np.float64)
    y = np.asfortranarray(y, dtype=np.float64)
    return _tikhonov_lambda2_power_rule(X, y, tFDR, tik_control, trex_control, int(seed),
                                        [float(v) for v in grid], float(knee_tolerance), verbose)


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
            ``gvs_ctrl.gvs_type`` must stay ``GVSType.EN`` (the default):
            T-Rex SPCA is defined on the elastic net, and ``GVSType.IEN``
            raises ``ValueError`` in ``select()``. The EN solver flavour is
            chosen via ``spca_ctrl.en_solver``. The GVS ridge penalty follows
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
    "normalize_design",
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
    "Lambda2CVDiagnostics",
    "TikLambda2Method",
    "PilotStatistic",
    "TikPenaltyStructure",
    "Lambda2PickInfo",
    "Lambda2PilotResult",
    "TRexTikhonovControlParameter",
    "TRexTikhonovSelector",
    "gamma_to_k",
    "build_group_centering_k",
    "Lambda2Probe",
    "Lambda2PowerRuleResult",
    "tikhonov_lambda2_power_rule",
    # Submodules
    "ml_methods",
    "tsolvers",
    "trex_selector_methods",
    "utils",
]
