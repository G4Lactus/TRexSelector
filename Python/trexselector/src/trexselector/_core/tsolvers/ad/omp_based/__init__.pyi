"""
OMP-based T-Solvers
"""
from __future__ import annotations
import numpy
import numpy.typing
import trexselector._core.tsolvers.ad
import typing
__all__: list[str] = ['FullyCorrective', 'LineSearch', 'NCGMPVariant', 'TACGP_Solver', 'TGP_Solver', 'TMP_Solver', 'TNCGMP_Solver', 'TOMP_Solver', 'TOOLS_Solver']
class NCGMPVariant:
    """
    Members:
    
      LineSearch : Computes updates using standard backtracking / exact line search.
    
      FullyCorrective : Computes fully corrective projections for all active set members at each step.
    """
    FullyCorrective: typing.ClassVar[NCGMPVariant]  # value = <NCGMPVariant.FullyCorrective: 1>
    LineSearch: typing.ClassVar[NCGMPVariant]  # value = <NCGMPVariant.LineSearch: 0>
    __members__: typing.ClassVar[dict[str, NCGMPVariant]]  # value = {'LineSearch': <NCGMPVariant.LineSearch: 0>, 'FullyCorrective': <NCGMPVariant.FullyCorrective: 1>}
    @typing.overload
    def __eq__(self, other: NCGMPVariant) -> bool:
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
    def __ne__(self, other: NCGMPVariant) -> bool:
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
class TACGP_Solver:
    """
    Terminating approximate conjugate gradient pursuit (T-ACGP) solver.
    
    References
    ----------
    T. Blumensath and M. E. Davies, "Gradient Pursuits," IEEE Trans. Signal Process.,
      vol. 56, no. 6, pp. 2370-2382, 2008.
    F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False
      Discovery Rate Control," in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026,
      pp. 2511-2515.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). 
        
        In-place contract: with normalize=True or intercept=True the solver is a WRITER. It centers/scales the columns of X and D IN PLACE through the views (a Fortran-ordered float64 array is shared, a C-ordered one is copied first), keeps the statistics itself and holds a claim on both buffers until restore() or destruction, both of which write X and D back (restore() is idempotent and must be given the same arrays); constructing a second writer (T-solver or T-Rex selector) on a claimed buffer raises RuntimeError, because a second in-place transformation moves the data by rounding. With normalize=False and intercept=False the solver is a READER: it runs as constructed and never touches the buffers. A pipeline with several solvers on one buffer should pre-scale once in front (LpNormScaler) and construct the solvers as readers; a solver resumed with load() is a reader that expects the buffers in the saved state. y is never written (the solver centers an internal copy).
        """
    def clearWarnings(self) -> None:
        """
        Clear the recorded warnings.
        """
    def executeStep(self, T_stop: typing.SupportsInt | typing.SupportsIndex = 0, early_stop: bool = True) -> None:
        """
        Executes the underlying algorithmic solve mechanics up to constraint `T_stop`. Pass 0 or -1 to calculate the full sequential path.
        """
    def getActions(self) -> list[list[int]]:
        """
        Get the integer codes for actions taken at each step. Entries follow the R lars convention: 1-based signed indices, +(j+1) for the addition and -(j+1) for the removal of 0-based variable j (decode with abs(a) - 1).
        """
    def getActiveDummyIndices(self) -> list[int]:
        """
        Get indices of active dummy variables at the current step.
        """
    def getActivePredictorIndices(self) -> list[int]:
        """
        Get indices of active predictor (non-dummy) variables at the current step.
        """
    def getActives(self) -> list[int]:
        """
        Get indices of variables that are active at the current step.
        """
    def getBeta(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Retrieves the active regression coefficients (Beta) for a specific step.
        """
    def getBetaPath(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Retrieves a historical path history block containing all calculated Beta coefficients across execution steps.
        """
    def getCp(self) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step.
        """
    def getCpWithSigma(self, sigma_hat_sq: typing.SupportsFloat | typing.SupportsIndex) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step with provided sigma squared.
        """
    def getDoF(self) -> list[int]:
        """
        Get the degrees of freedom at each step.
        """
    def getDroppedIndices(self) -> list[int]:
        """
        Get indices of variables that were dropped during the execution.
        """
    def getDummyStartIndex(self) -> int:
        """
        Find where dummy variables start within the merged predictor index space.
        """
    def getInactives(self) -> list[int]:
        """
        Get indices of variables that are inactive at the current step.
        """
    def getIntercept(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> float:
        """
        Retrieves the calculated model intercept for a specific step.
        """
    def getMemoryBytes(self) -> int:
        """
        Heap bytes the solver holds: its copy of y, the residual, correlations, Cholesky factor, coefficient path, diagnostics and reserved capacity. X and D are borrowed and not counted, nor are the temporaries inside a step, so the peak of a step is higher.
        """
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumSteps(self) -> int:
        """
        Quantify the number of path steps successfully taken thus far.
        """
    def getNumThreads(self) -> int:
        """
        The resolved thread budget of the solver's own passes (at least 1).
        """
    def getR2(self) -> list[float]:
        """
        Get the coefficient of determination (R^2) at each step.
        """
    def getRSS(self) -> list[float]:
        """
        Get the residual sum of squares at each step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isConnected(self) -> bool:
        """
        Check whether the solver has mapped access to data predictors.
        """
    def load(self, filepath: str) -> None:
        """
        Hydrate a solver explicitly from a pre-calculated Cereal output file on disk.
        """
    def reconnect(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        """
        Reconnect the underlying native solver engine to a new batch of underlying Matrices.
        """
    def restore(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"]) -> None:
        """
        Undo the solver's in-place centering/scaling of X and D (a writer, see the constructor) and release its claim on the buffers, so that a later writer may take them; y is not modified. No-op for a reader.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
        """
    def setExchangeableTie(self, alpha: typing.SupportsFloat | typing.SupportsIndex, floor: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> None:
        """
        Configure exchangeability-calibrated stochastic tie-breaking for greedy candidate selection (TOMP/TAFS); alpha <= 0 turns it off.
        """
    def setNumThreads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the solver's own passes over the design: 1 serial, a whole number of at least 1 (other values raise). A pass splits only above about a million entries and never inside a parallel region of the caller.
        """
    def setTieSeed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the random seed for reproducible tie-breaking among dummy candidates.
        """
    def setTolerance(self, tol: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the numerical tolerance used for internal numeric comparisons.
        """
    def solverTypeToString(self) -> str:
        """
        Returns a string descriptor mapping for the underlying wrapped T-Solver.
        """
class TGP_Solver:
    """
    Terminating gradient pursuit (T-GP) solver.
    
    References
    ----------
    T. Blumensath and M. E. Davies, "Gradient Pursuits," IEEE Trans. Signal Process.,
      vol. 56, no. 6, pp. 2370-2382, 2008.
    F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False
      Discovery Rate Control," in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026,
      pp. 2511-2515.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). 
        
        In-place contract: with normalize=True or intercept=True the solver is a WRITER. It centers/scales the columns of X and D IN PLACE through the views (a Fortran-ordered float64 array is shared, a C-ordered one is copied first), keeps the statistics itself and holds a claim on both buffers until restore() or destruction, both of which write X and D back (restore() is idempotent and must be given the same arrays); constructing a second writer (T-solver or T-Rex selector) on a claimed buffer raises RuntimeError, because a second in-place transformation moves the data by rounding. With normalize=False and intercept=False the solver is a READER: it runs as constructed and never touches the buffers. A pipeline with several solvers on one buffer should pre-scale once in front (LpNormScaler) and construct the solvers as readers; a solver resumed with load() is a reader that expects the buffers in the saved state. y is never written (the solver centers an internal copy).
        """
    def clearWarnings(self) -> None:
        """
        Clear the recorded warnings.
        """
    def executeStep(self, T_stop: typing.SupportsInt | typing.SupportsIndex = 0, early_stop: bool = True) -> None:
        """
        Executes the underlying algorithmic solve mechanics up to constraint `T_stop`. Pass 0 or -1 to calculate the full sequential path.
        """
    def getActions(self) -> list[list[int]]:
        """
        Get the integer codes for actions taken at each step. Entries follow the R lars convention: 1-based signed indices, +(j+1) for the addition and -(j+1) for the removal of 0-based variable j (decode with abs(a) - 1).
        """
    def getActiveDummyIndices(self) -> list[int]:
        """
        Get indices of active dummy variables at the current step.
        """
    def getActivePredictorIndices(self) -> list[int]:
        """
        Get indices of active predictor (non-dummy) variables at the current step.
        """
    def getActives(self) -> list[int]:
        """
        Get indices of variables that are active at the current step.
        """
    def getBeta(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Retrieves the active regression coefficients (Beta) for a specific step.
        """
    def getBetaPath(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Retrieves a historical path history block containing all calculated Beta coefficients across execution steps.
        """
    def getCp(self) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step.
        """
    def getCpWithSigma(self, sigma_hat_sq: typing.SupportsFloat | typing.SupportsIndex) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step with provided sigma squared.
        """
    def getDoF(self) -> list[int]:
        """
        Get the degrees of freedom at each step.
        """
    def getDroppedIndices(self) -> list[int]:
        """
        Get indices of variables that were dropped during the execution.
        """
    def getDummyStartIndex(self) -> int:
        """
        Find where dummy variables start within the merged predictor index space.
        """
    def getInactives(self) -> list[int]:
        """
        Get indices of variables that are inactive at the current step.
        """
    def getIntercept(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> float:
        """
        Retrieves the calculated model intercept for a specific step.
        """
    def getMemoryBytes(self) -> int:
        """
        Heap bytes the solver holds: its copy of y, the residual, correlations, Cholesky factor, coefficient path, diagnostics and reserved capacity. X and D are borrowed and not counted, nor are the temporaries inside a step, so the peak of a step is higher.
        """
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumSteps(self) -> int:
        """
        Quantify the number of path steps successfully taken thus far.
        """
    def getNumThreads(self) -> int:
        """
        The resolved thread budget of the solver's own passes (at least 1).
        """
    def getR2(self) -> list[float]:
        """
        Get the coefficient of determination (R^2) at each step.
        """
    def getRSS(self) -> list[float]:
        """
        Get the residual sum of squares at each step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isConnected(self) -> bool:
        """
        Check whether the solver has mapped access to data predictors.
        """
    def load(self, filepath: str) -> None:
        """
        Hydrate a solver explicitly from a pre-calculated Cereal output file on disk.
        """
    def reconnect(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        """
        Reconnect the underlying native solver engine to a new batch of underlying Matrices.
        """
    def restore(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"]) -> None:
        """
        Undo the solver's in-place centering/scaling of X and D (a writer, see the constructor) and release its claim on the buffers, so that a later writer may take them; y is not modified. No-op for a reader.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
        """
    def setExchangeableTie(self, alpha: typing.SupportsFloat | typing.SupportsIndex, floor: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> None:
        """
        Configure exchangeability-calibrated stochastic tie-breaking for greedy candidate selection (TOMP/TAFS); alpha <= 0 turns it off.
        """
    def setNumThreads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the solver's own passes over the design: 1 serial, a whole number of at least 1 (other values raise). A pass splits only above about a million entries and never inside a parallel region of the caller.
        """
    def setTieSeed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the random seed for reproducible tie-breaking among dummy candidates.
        """
    def setTolerance(self, tol: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the numerical tolerance used for internal numeric comparisons.
        """
    def solverTypeToString(self) -> str:
        """
        Returns a string descriptor mapping for the underlying wrapped T-Solver.
        """
class TMP_Solver:
    """
    Terminating matching pursuit (T-MP) solver.
    
    References
    ----------
    S. G. Mallat and Z. Zhang, "Matching Pursuits with Time-Frequency Dictionaries," IEEE
      Trans. Signal Process., vol. 41, no. 12, pp. 3397-3415, 1993, doi: 10.1109/78.258082.
    F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False
      Discovery Rate Control," in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026,
      pp. 2511-2515.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). 
        
        In-place contract: with normalize=True or intercept=True the solver is a WRITER. It centers/scales the columns of X and D IN PLACE through the views (a Fortran-ordered float64 array is shared, a C-ordered one is copied first), keeps the statistics itself and holds a claim on both buffers until restore() or destruction, both of which write X and D back (restore() is idempotent and must be given the same arrays); constructing a second writer (T-solver or T-Rex selector) on a claimed buffer raises RuntimeError, because a second in-place transformation moves the data by rounding. With normalize=False and intercept=False the solver is a READER: it runs as constructed and never touches the buffers. A pipeline with several solvers on one buffer should pre-scale once in front (LpNormScaler) and construct the solvers as readers; a solver resumed with load() is a reader that expects the buffers in the saved state. y is never written (the solver centers an internal copy).
        """
    def clearWarnings(self) -> None:
        """
        Clear the recorded warnings.
        """
    def executeStep(self, T_stop: typing.SupportsInt | typing.SupportsIndex = 0, early_stop: bool = True) -> None:
        """
        Executes the underlying algorithmic solve mechanics up to constraint `T_stop`. Pass 0 or -1 to calculate the full sequential path.
        """
    def getActions(self) -> list[list[int]]:
        """
        Get the integer codes for actions taken at each step. Entries follow the R lars convention: 1-based signed indices, +(j+1) for the addition and -(j+1) for the removal of 0-based variable j (decode with abs(a) - 1).
        """
    def getActiveDummyIndices(self) -> list[int]:
        """
        Get indices of active dummy variables at the current step.
        """
    def getActivePredictorIndices(self) -> list[int]:
        """
        Get indices of active predictor (non-dummy) variables at the current step.
        """
    def getActives(self) -> list[int]:
        """
        Get indices of variables that are active at the current step.
        """
    def getBeta(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Retrieves the active regression coefficients (Beta) for a specific step.
        """
    def getBetaPath(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Retrieves a historical path history block containing all calculated Beta coefficients across execution steps.
        """
    def getCp(self) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step.
        """
    def getCpWithSigma(self, sigma_hat_sq: typing.SupportsFloat | typing.SupportsIndex) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step with provided sigma squared.
        """
    def getDoF(self) -> list[int]:
        """
        Get the degrees of freedom at each step.
        """
    def getDroppedIndices(self) -> list[int]:
        """
        Get indices of variables that were dropped during the execution.
        """
    def getDummyStartIndex(self) -> int:
        """
        Find where dummy variables start within the merged predictor index space.
        """
    def getInactives(self) -> list[int]:
        """
        Get indices of variables that are inactive at the current step.
        """
    def getIntercept(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> float:
        """
        Retrieves the calculated model intercept for a specific step.
        """
    def getMemoryBytes(self) -> int:
        """
        Heap bytes the solver holds: its copy of y, the residual, correlations, Cholesky factor, coefficient path, diagnostics and reserved capacity. X and D are borrowed and not counted, nor are the temporaries inside a step, so the peak of a step is higher.
        """
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumSteps(self) -> int:
        """
        Quantify the number of path steps successfully taken thus far.
        """
    def getNumThreads(self) -> int:
        """
        The resolved thread budget of the solver's own passes (at least 1).
        """
    def getR2(self) -> list[float]:
        """
        Get the coefficient of determination (R^2) at each step.
        """
    def getRSS(self) -> list[float]:
        """
        Get the residual sum of squares at each step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isConnected(self) -> bool:
        """
        Check whether the solver has mapped access to data predictors.
        """
    def load(self, filepath: str) -> None:
        """
        Hydrate a solver explicitly from a pre-calculated Cereal output file on disk.
        """
    def reconnect(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        """
        Reconnect the underlying native solver engine to a new batch of underlying Matrices.
        """
    def restore(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"]) -> None:
        """
        Undo the solver's in-place centering/scaling of X and D (a writer, see the constructor) and release its claim on the buffers, so that a later writer may take them; y is not modified. No-op for a reader.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
        """
    def setExchangeableTie(self, alpha: typing.SupportsFloat | typing.SupportsIndex, floor: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> None:
        """
        Configure exchangeability-calibrated stochastic tie-breaking for greedy candidate selection (TOMP/TAFS); alpha <= 0 turns it off.
        """
    def setNumThreads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the solver's own passes over the design: 1 serial, a whole number of at least 1 (other values raise). A pass splits only above about a million entries and never inside a parallel region of the caller.
        """
    def setTieSeed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the random seed for reproducible tie-breaking among dummy candidates.
        """
    def setTolerance(self, tol: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the numerical tolerance used for internal numeric comparisons.
        """
    def solverTypeToString(self) -> str:
        """
        Returns a string descriptor mapping for the underlying wrapped T-Solver.
        """
class TNCGMP_Solver:
    """
    Terminating norm-corrective generalized matching pursuit (T-NCGMP) solver.
    
    References
    ----------
    F. Locatello, R. Khanna, M. Tschannen, and M. Jaggi, "A Unified Optimization View on
      Generalized Matching Pursuit and Frank-Wolfe," in Proc. 20th Int. Conf. Artif. Intell.
      Statist. (AISTATS), JMLR: W&CP vol. 54, 2017.
    F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False
      Discovery Rate Control," in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026,
      pp. 2511-2515.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). Reads options.ncgmp_variant (LineSearch or FullyCorrective).
        
        In-place contract: with normalize=True or intercept=True the solver is a WRITER. It centers/scales the columns of X and D IN PLACE through the views (a Fortran-ordered float64 array is shared, a C-ordered one is copied first), keeps the statistics itself and holds a claim on both buffers until restore() or destruction, both of which write X and D back (restore() is idempotent and must be given the same arrays); constructing a second writer (T-solver or T-Rex selector) on a claimed buffer raises RuntimeError, because a second in-place transformation moves the data by rounding. With normalize=False and intercept=False the solver is a READER: it runs as constructed and never touches the buffers. A pipeline with several solvers on one buffer should pre-scale once in front (LpNormScaler) and construct the solvers as readers; a solver resumed with load() is a reader that expects the buffers in the saved state. y is never written (the solver centers an internal copy).
        """
    def clearWarnings(self) -> None:
        """
        Clear the recorded warnings.
        """
    def executeStep(self, T_stop: typing.SupportsInt | typing.SupportsIndex = 0, early_stop: bool = True) -> None:
        """
        Executes the underlying algorithmic solve mechanics up to constraint `T_stop`. Pass 0 or -1 to calculate the full sequential path.
        """
    def getActions(self) -> list[list[int]]:
        """
        Get the integer codes for actions taken at each step. Entries follow the R lars convention: 1-based signed indices, +(j+1) for the addition and -(j+1) for the removal of 0-based variable j (decode with abs(a) - 1).
        """
    def getActiveDummyIndices(self) -> list[int]:
        """
        Get indices of active dummy variables at the current step.
        """
    def getActivePredictorIndices(self) -> list[int]:
        """
        Get indices of active predictor (non-dummy) variables at the current step.
        """
    def getActives(self) -> list[int]:
        """
        Get indices of variables that are active at the current step.
        """
    def getBeta(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Retrieves the active regression coefficients (Beta) for a specific step.
        """
    def getBetaPath(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Retrieves a historical path history block containing all calculated Beta coefficients across execution steps.
        """
    def getCp(self) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step.
        """
    def getCpWithSigma(self, sigma_hat_sq: typing.SupportsFloat | typing.SupportsIndex) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step with provided sigma squared.
        """
    def getDoF(self) -> list[int]:
        """
        Get the degrees of freedom at each step.
        """
    def getDroppedIndices(self) -> list[int]:
        """
        Get indices of variables that were dropped during the execution.
        """
    def getDummyStartIndex(self) -> int:
        """
        Find where dummy variables start within the merged predictor index space.
        """
    def getInactives(self) -> list[int]:
        """
        Get indices of variables that are inactive at the current step.
        """
    def getIntercept(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> float:
        """
        Retrieves the calculated model intercept for a specific step.
        """
    def getMemoryBytes(self) -> int:
        """
        Heap bytes the solver holds: its copy of y, the residual, correlations, Cholesky factor, coefficient path, diagnostics and reserved capacity. X and D are borrowed and not counted, nor are the temporaries inside a step, so the peak of a step is higher.
        """
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumSteps(self) -> int:
        """
        Quantify the number of path steps successfully taken thus far.
        """
    def getNumThreads(self) -> int:
        """
        The resolved thread budget of the solver's own passes (at least 1).
        """
    def getR2(self) -> list[float]:
        """
        Get the coefficient of determination (R^2) at each step.
        """
    def getRSS(self) -> list[float]:
        """
        Get the residual sum of squares at each step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isConnected(self) -> bool:
        """
        Check whether the solver has mapped access to data predictors.
        """
    def load(self, filepath: str) -> None:
        """
        Hydrate a solver explicitly from a pre-calculated Cereal output file on disk.
        """
    def reconnect(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        """
        Reconnect the underlying native solver engine to a new batch of underlying Matrices.
        """
    def restore(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"]) -> None:
        """
        Undo the solver's in-place centering/scaling of X and D (a writer, see the constructor) and release its claim on the buffers, so that a later writer may take them; y is not modified. No-op for a reader.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
        """
    def setExchangeableTie(self, alpha: typing.SupportsFloat | typing.SupportsIndex, floor: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> None:
        """
        Configure exchangeability-calibrated stochastic tie-breaking for greedy candidate selection (TOMP/TAFS); alpha <= 0 turns it off.
        """
    def setNumThreads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the solver's own passes over the design: 1 serial, a whole number of at least 1 (other values raise). A pass splits only above about a million entries and never inside a parallel region of the caller.
        """
    def setTieSeed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the random seed for reproducible tie-breaking among dummy candidates.
        """
    def setTolerance(self, tol: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the numerical tolerance used for internal numeric comparisons.
        """
    def solverTypeToString(self) -> str:
        """
        Returns a string descriptor mapping for the underlying wrapped T-Solver.
        """
class TOMP_Solver:
    """
    Terminating orthogonal matching pursuit (T-OMP) solver.
    
    References
    ----------
    Y. C. Pati, R. Rezaiifar, and P. S. Krishnaprasad, "Orthogonal Matching Pursuit:
      Recursive Function Approximation with Applications to Wavelet Decomposition," in Proc.
      27th Asilomar Conf. Signals, Syst., Comput., 1993, pp. 40-44.
    J. A. Tropp, "Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans.
      Inf. Theory, vol. 50, no. 10, pp. 2231-2242, 2004, doi: 10.1109/TIT.2004.834793.
    T. Blumensath and M. E. Davies, "On the Difference Between Orthogonal Matching Pursuit
      and Orthogonal Least Squares," unpublished note, draft of March 29, 2007.
    F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False
      Discovery Rate Control," in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026,
      pp. 2511-2515.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). Greedy: reads options.exch_tie_alpha / exch_tie_floor.
        
        In-place contract: with normalize=True or intercept=True the solver is a WRITER. It centers/scales the columns of X and D IN PLACE through the views (a Fortran-ordered float64 array is shared, a C-ordered one is copied first), keeps the statistics itself and holds a claim on both buffers until restore() or destruction, both of which write X and D back (restore() is idempotent and must be given the same arrays); constructing a second writer (T-solver or T-Rex selector) on a claimed buffer raises RuntimeError, because a second in-place transformation moves the data by rounding. With normalize=False and intercept=False the solver is a READER: it runs as constructed and never touches the buffers. A pipeline with several solvers on one buffer should pre-scale once in front (LpNormScaler) and construct the solvers as readers; a solver resumed with load() is a reader that expects the buffers in the saved state. y is never written (the solver centers an internal copy).
        """
    def clearWarnings(self) -> None:
        """
        Clear the recorded warnings.
        """
    def executeStep(self, T_stop: typing.SupportsInt | typing.SupportsIndex = 0, early_stop: bool = True) -> None:
        """
        Executes the underlying algorithmic solve mechanics up to constraint `T_stop`. Pass 0 or -1 to calculate the full sequential path.
        """
    def getActions(self) -> list[list[int]]:
        """
        Get the integer codes for actions taken at each step. Entries follow the R lars convention: 1-based signed indices, +(j+1) for the addition and -(j+1) for the removal of 0-based variable j (decode with abs(a) - 1).
        """
    def getActiveDummyIndices(self) -> list[int]:
        """
        Get indices of active dummy variables at the current step.
        """
    def getActivePredictorIndices(self) -> list[int]:
        """
        Get indices of active predictor (non-dummy) variables at the current step.
        """
    def getActives(self) -> list[int]:
        """
        Get indices of variables that are active at the current step.
        """
    def getBeta(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Retrieves the active regression coefficients (Beta) for a specific step.
        """
    def getBetaPath(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Retrieves a historical path history block containing all calculated Beta coefficients across execution steps.
        """
    def getCp(self) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step.
        """
    def getCpWithSigma(self, sigma_hat_sq: typing.SupportsFloat | typing.SupportsIndex) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step with provided sigma squared.
        """
    def getDoF(self) -> list[int]:
        """
        Get the degrees of freedom at each step.
        """
    def getDroppedIndices(self) -> list[int]:
        """
        Get indices of variables that were dropped during the execution.
        """
    def getDummyStartIndex(self) -> int:
        """
        Find where dummy variables start within the merged predictor index space.
        """
    def getInactives(self) -> list[int]:
        """
        Get indices of variables that are inactive at the current step.
        """
    def getIntercept(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> float:
        """
        Retrieves the calculated model intercept for a specific step.
        """
    def getMemoryBytes(self) -> int:
        """
        Heap bytes the solver holds: its copy of y, the residual, correlations, Cholesky factor, coefficient path, diagnostics and reserved capacity. X and D are borrowed and not counted, nor are the temporaries inside a step, so the peak of a step is higher.
        """
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumSteps(self) -> int:
        """
        Quantify the number of path steps successfully taken thus far.
        """
    def getNumThreads(self) -> int:
        """
        The resolved thread budget of the solver's own passes (at least 1).
        """
    def getR2(self) -> list[float]:
        """
        Get the coefficient of determination (R^2) at each step.
        """
    def getRSS(self) -> list[float]:
        """
        Get the residual sum of squares at each step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isConnected(self) -> bool:
        """
        Check whether the solver has mapped access to data predictors.
        """
    def load(self, filepath: str) -> None:
        """
        Hydrate a solver explicitly from a pre-calculated Cereal output file on disk.
        """
    def reconnect(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        """
        Reconnect the underlying native solver engine to a new batch of underlying Matrices.
        """
    def restore(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"]) -> None:
        """
        Undo the solver's in-place centering/scaling of X and D (a writer, see the constructor) and release its claim on the buffers, so that a later writer may take them; y is not modified. No-op for a reader.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
        """
    def setExchangeableTie(self, alpha: typing.SupportsFloat | typing.SupportsIndex, floor: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> None:
        """
        Configure exchangeability-calibrated stochastic tie-breaking for greedy candidate selection (TOMP/TAFS); alpha <= 0 turns it off.
        """
    def setNumThreads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the solver's own passes over the design: 1 serial, a whole number of at least 1 (other values raise). A pass splits only above about a million entries and never inside a parallel region of the caller.
        """
    def setTieSeed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the random seed for reproducible tie-breaking among dummy candidates.
        """
    def setTolerance(self, tol: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the numerical tolerance used for internal numeric comparisons.
        """
    def solverTypeToString(self) -> str:
        """
        Returns a string descriptor mapping for the underlying wrapped T-Solver.
        """
class TOOLS_Solver:
    """
    Terminating orthogonal least squares (T-OLS) solver.
    
    References
    ----------
    S. Chen, S. A. Billings, and W. Luo, "Orthogonal Least Squares Methods and Their
      Application to Non-Linear System Identification," Int. J. Control, vol. 50, no. 5,
      pp. 1873-1896, 1989, doi: 10.1080/00207178908953472.
    T. Blumensath and M. E. Davies, "On the Difference Between Orthogonal Matching Pursuit
      and Orthogonal Least Squares," unpublished note, draft of March 29, 2007.
    F. Scheidt and M. Muma, "Terminating Greedy Pursuits for High-Dimensional False
      Discovery Rate Control," in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2026,
      pp. 2511-2515.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). 
        
        In-place contract: with normalize=True or intercept=True the solver is a WRITER. It centers/scales the columns of X and D IN PLACE through the views (a Fortran-ordered float64 array is shared, a C-ordered one is copied first), keeps the statistics itself and holds a claim on both buffers until restore() or destruction, both of which write X and D back (restore() is idempotent and must be given the same arrays); constructing a second writer (T-solver or T-Rex selector) on a claimed buffer raises RuntimeError, because a second in-place transformation moves the data by rounding. With normalize=False and intercept=False the solver is a READER: it runs as constructed and never touches the buffers. A pipeline with several solvers on one buffer should pre-scale once in front (LpNormScaler) and construct the solvers as readers; a solver resumed with load() is a reader that expects the buffers in the saved state. y is never written (the solver centers an internal copy).
        """
    def clearWarnings(self) -> None:
        """
        Clear the recorded warnings.
        """
    def executeStep(self, T_stop: typing.SupportsInt | typing.SupportsIndex = 0, early_stop: bool = True) -> None:
        """
        Executes the underlying algorithmic solve mechanics up to constraint `T_stop`. Pass 0 or -1 to calculate the full sequential path.
        """
    def getActions(self) -> list[list[int]]:
        """
        Get the integer codes for actions taken at each step. Entries follow the R lars convention: 1-based signed indices, +(j+1) for the addition and -(j+1) for the removal of 0-based variable j (decode with abs(a) - 1).
        """
    def getActiveDummyIndices(self) -> list[int]:
        """
        Get indices of active dummy variables at the current step.
        """
    def getActivePredictorIndices(self) -> list[int]:
        """
        Get indices of active predictor (non-dummy) variables at the current step.
        """
    def getActives(self) -> list[int]:
        """
        Get indices of variables that are active at the current step.
        """
    def getBeta(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Retrieves the active regression coefficients (Beta) for a specific step.
        """
    def getBetaPath(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Retrieves a historical path history block containing all calculated Beta coefficients across execution steps.
        """
    def getCp(self) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step.
        """
    def getCpWithSigma(self, sigma_hat_sq: typing.SupportsFloat | typing.SupportsIndex) -> list[float]:
        """
        Get the estimated Mallow's Cp values at each step with provided sigma squared.
        """
    def getDoF(self) -> list[int]:
        """
        Get the degrees of freedom at each step.
        """
    def getDroppedIndices(self) -> list[int]:
        """
        Get indices of variables that were dropped during the execution.
        """
    def getDummyStartIndex(self) -> int:
        """
        Find where dummy variables start within the merged predictor index space.
        """
    def getInactives(self) -> list[int]:
        """
        Get indices of variables that are inactive at the current step.
        """
    def getIntercept(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> float:
        """
        Retrieves the calculated model intercept for a specific step.
        """
    def getMemoryBytes(self) -> int:
        """
        Heap bytes the solver holds: its copy of y, the residual, correlations, Cholesky factor, coefficient path, diagnostics and reserved capacity. X and D are borrowed and not counted, nor are the temporaries inside a step, so the peak of a step is higher.
        """
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumSteps(self) -> int:
        """
        Quantify the number of path steps successfully taken thus far.
        """
    def getNumThreads(self) -> int:
        """
        The resolved thread budget of the solver's own passes (at least 1).
        """
    def getR2(self) -> list[float]:
        """
        Get the coefficient of determination (R^2) at each step.
        """
    def getRSS(self) -> list[float]:
        """
        Get the residual sum of squares at each step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isConnected(self) -> bool:
        """
        Check whether the solver has mapped access to data predictors.
        """
    def load(self, filepath: str) -> None:
        """
        Hydrate a solver explicitly from a pre-calculated Cereal output file on disk.
        """
    def reconnect(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        """
        Reconnect the underlying native solver engine to a new batch of underlying Matrices.
        """
    def restore(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"]) -> None:
        """
        Undo the solver's in-place centering/scaling of X and D (a writer, see the constructor) and release its claim on the buffers, so that a later writer may take them; y is not modified. No-op for a reader.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
        """
    def setExchangeableTie(self, alpha: typing.SupportsFloat | typing.SupportsIndex, floor: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> None:
        """
        Configure exchangeability-calibrated stochastic tie-breaking for greedy candidate selection (TOMP/TAFS); alpha <= 0 turns it off.
        """
    def setNumThreads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the solver's own passes over the design: 1 serial, a whole number of at least 1 (other values raise). A pass splits only above about a million entries and never inside a parallel region of the caller.
        """
    def setTieSeed(self, seed: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the random seed for reproducible tie-breaking among dummy candidates.
        """
    def setTolerance(self, tol: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the numerical tolerance used for internal numeric comparisons.
        """
    def solverTypeToString(self) -> str:
        """
        Returns a string descriptor mapping for the underlying wrapped T-Solver.
        """
FullyCorrective: NCGMPVariant  # value = <NCGMPVariant.FullyCorrective: 1>
LineSearch: NCGMPVariant  # value = <NCGMPVariant.LineSearch: 0>
