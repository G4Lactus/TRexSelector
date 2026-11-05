"""
LARS-based T-Solvers
"""
from __future__ import annotations
import numpy
import numpy.typing
import trexselector._core.tsolvers.ad
import typing
__all__: list[str] = ['TENETAug_Solver', 'TENET_Solver', 'TIENETAug_Solver', 'TIENET_Solver', 'TIKNET_Solver', 'TLARS_Solver', 'TLASSO_Solver', 'TSTAGEWISE_Solver', 'TSTEPWISE_Solver']
class TENETAug_Solver:
    """
    Terminating elastic net (T-ENETAug) solver via row augmentation around an inner T-LASSO.
    
    References
    ----------
    H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
    B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann.
      Statist., vol. 32, no. 2, pp. 407-499, 2004.
    J. Machkour, M. Muma, and D. P. Palomar, "False Discovery Rate Control for Grouped
      Variable Selection in High-Dimensional Linear Models Using the T-Knock Filter," in
      Proc. 30th Eur. Signal Process. Conf. (EUSIPCO), 2022, pp. 892-896.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). Reads options.lambda2 and options.tenet_aug_use_lars (pure-LARS inner path).
        
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
class TENET_Solver:
    """
    Terminating elastic net (T-ENET) solver via the native T-LARS-EN path (L1 + L2 penalty).
    
    References
    ----------
    H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
    B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann.
      Statist., vol. 32, no. 2, pp. 407-499, 2004.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). Reads options.lambda2, the secondary L2 penalty (default 0: the lasso path).
        
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
    def getAlpha(self) -> float:
        """
        Get the Elastic Net mixing parameter alpha implied by lambda2.
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
    def getCyclingRatio(self) -> float:
        """
        Get the cycling ratio (removals / additions).
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
    def getNumRemovals(self) -> int:
        """
        Get the total number of variable removals during solution path.
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
class TIENETAug_Solver:
    """
    Terminating informed elastic net (T-IENETAug) solver via row augmentation around an inner
    T-LASSO.
    
    References
    ----------
    J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped
      Variable Selection and FDR Control in Genomics Research," in Proc. IEEE 9th Int.
      Workshop Comput. Adv. Multi-Sensor Adapt. Process. (CAMSAP), 2023, pp. 466-470.
    H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
    B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann.
      Statist., vol. 32, no. 2, pp. 407-499, 2004.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], groups: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 1]"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the augmented-LASSO Informed Elastic Net solver with a 0-based group id per original variable (`groups`, length p); reads options.lambda2 (the group-ridge penalty) and options.tenet_aug_use_lars.
        
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
    def getGroups(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"]:
        """
        Get the 0-based group id per original variable (length p).
        """
    def getInactives(self) -> list[int]:
        """
        Get indices of variables that are inactive at the current step.
        """
    def getIntercept(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> float:
        """
        Retrieves the calculated model intercept for a specific step.
        """
    def getLambda2(self) -> float:
        """
        Get the group-ridge penalty lambda2 used.
        """
    def getMemoryBytes(self) -> int:
        """
        Heap bytes the solver holds: its copy of y, the residual, correlations, Cholesky factor, coefficient path, diagnostics and reserved capacity. X and D are borrowed and not counted, nor are the temporaries inside a step, so the peak of a step is higher.
        """
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumGroups(self) -> int:
        """
        Get the number of disjoint groups.
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
class TIENET_Solver:
    """
    Terminating informed elastic net (T-IENET) solver via the native pathwise LARS-IEN algorithm.
    
    References
    ----------
    J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped
      Variable Selection and FDR Control in Genomics Research," in Proc. IEEE 9th Int.
      Workshop Comput. Adv. Multi-Sensor Adapt. Process. (CAMSAP), 2023, pp. 466-470.
    H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
    B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann.
      Statist., vol. 32, no. 2, pp. 407-499, 2004.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], groups: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 1]"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y with a 0-based group id per original variable (`groups`, length p); every other input comes from `options` (SolverOptions). Reads options.lambda2, the group-ridge penalty.
        
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
    def getGroups(self) -> typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"]:
        """
        Get the 0-based group id per original variable (length p).
        """
    def getInactives(self) -> list[int]:
        """
        Get indices of variables that are inactive at the current step.
        """
    def getIntercept(self, step: typing.SupportsInt | typing.SupportsIndex = -1) -> float:
        """
        Retrieves the calculated model intercept for a specific step.
        """
    def getLambda(self) -> list[float]:
        """
        Get the regularization parameter sequence (max correlation per step).
        """
    def getLambda2(self) -> float:
        """
        Get the group-ridge penalty lambda2 used.
        """
    def getMemoryBytes(self) -> int:
        """
        Heap bytes the solver holds: its copy of y, the residual, correlations, Cholesky factor, coefficient path, diagnostics and reserved capacity. X and D are borrowed and not counted, nor are the temporaries inside a step, so the peak of a step is higher.
        """
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumGroups(self) -> int:
        """
        Get the number of disjoint groups.
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
class TIKNET_Solver:
    """
    Terminating Tikhonov net (T-TIKNET) solver via the native pathwise LARS-TIK algorithm.
    
    References
    ----------
    A. N. Tikhonov, "Solution of Incorrectly Formulated Problems and the Regularization
      Method," Soviet Math. Dokl., vol. 4, pp. 1035-1038, 1963 (English translation of Dokl.
      Akad. Nauk SSSR 151 (1963), no. 3, 501-504).
    Y. Park, L. Reichel, G. Rodriguez, and X. Yu, "Parameter Determination for Tikhonov
      Regularization Problems in General Form," J. Comput. Appl. Math., vol. 343, pp. 12-25,
      2018, doi: 10.1016/j.cam.2018.04.049.
    A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal
      Problems," Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article
      in the Special 40th Anniversary Issue).
    Y. Miche, M. van Heeswijk, P. Bas, O. Simula, and A. Lendasse, "TROP-ELM: A
      Double-Regularized ELM Using LARS and Tikhonov Regularization," Neurocomputing,
      vol. 74, pp. 2413-2421, 2011, doi: 10.1016/j.neucom.2010.12.042.
    B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann.
      Statist., vol. 32, no. 2, pp. 407-499, 2004.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], tikhonov_K: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y with the p x p PSD Tikhonov matrix K (a dense array; converted to sparse internally); every other input comes from `options` (SolverOptions). Reads options.lambda2, the quadratic penalty weight; FOLDED dummy coupling only (D.cols() a multiple of X.cols() when lambda2 > 0).
        
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
    def getLambda(self) -> list[float]:
        """
        Get the regularization parameter sequence (max correlation per step).
        """
    def getLambda2(self) -> float:
        """
        Get the quadratic penalty weight lambda2 used.
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
    def getTikhonovK(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        The Tikhonov matrix K as a dense p x p array.
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
class TLARS_Solver:
    """
    Terminating least angle regression (T-LARS) solver.
    
    References
    ----------
    B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann.
      Statist., vol. 32, no. 2, pp. 407-499, 2004.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). Plain T-LARS reads no family hyperparameter.
        
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
    def getKahanRefreshInterval(self) -> int:
        """
        Get the current Kahan summation refresh interval.
        """
    def getLambda(self) -> list[float]:
        """
        Get the regularization parameter sequence (max correlation per step).
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
    def setKahanRefreshInterval(self, interval: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the Kahan summation refresh interval (call before executeStep).
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
class TLASSO_Solver:
    """
    Terminating LASSO (T-LASSO) solver: the LARS-LASSO path with variable removals.
    
    References
    ----------
    R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," J. R. Stat. Soc. Ser.
      B Stat. Methodol., vol. 58, no. 1, pp. 267-288, 1996.
    B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann.
      Statist., vol. 32, no. 2, pp. 407-499, 2004.
    M. R. Osborne, B. Presnell, and B. A. Turlach, "On the LASSO and Its Dual," J. Comput.
      Graph. Statist., vol. 9, no. 2, pp. 319-337, 2000.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
    def getCyclingRatio(self) -> float:
        """
        Get the cycling ratio (removals / additions).
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
    def getNumRemovals(self) -> int:
        """
        Get the total number of variable removals during solution path.
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
class TSTAGEWISE_Solver:
    """
    Terminating incremental forward stagewise (T-Stagewise) solver via the LARS-stagewise path.
    
    References
    ----------
    B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann.
      Statist., vol. 32, no. 2, pp. 407-499, 2004.
    T. Hastie, J. Taylor, R. Tibshirani, and G. Walther, "Forward Stagewise Regression and
      the Monotone Lasso," Electron. J. Stat., vol. 1, pp. 1-29, 2007,
      doi: 10.1214/07-EJS004.
    R. J. Tibshirani, "A General Framework for Fast Stagewise Algorithms," J. Mach. Learn.
      Res., vol. 16, pp. 2543-2588, 2015.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
    def getCyclingRatio(self) -> float:
        """
        Get the cycling ratio (removals / additions).
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
    def getNumRemovals(self) -> int:
        """
        Get the total number of coefficient removals.
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
class TSTEPWISE_Solver:
    """
    Terminating forward stepwise regression (T-Stepwise) solver.
    
    References
    ----------
    B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann.
      Statist., vol. 32, no. 2, pp. 407-499, 2004.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
