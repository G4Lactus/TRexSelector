"""
Coordinate-descent T-Solvers
"""
from __future__ import annotations
import numpy
import numpy.typing
import trexselector._core.tsolvers.ad
import typing
__all__: list[str] = ['TCDENET_Solver', 'TCDIENET_Solver', 'TCDLASSO_Solver', 'TCDTIKNET_Solver']
class TCDENET_Solver:
    """
    Terminating cyclic coordinate descent elastic net (T-CDENET) solver.
    
    References
    ----------
    H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
    J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate
      Optimization," Ann. Appl. Stat., vol. 1, no. 2, pp. 302-332, 2007,
      doi: 10.1214/07-AOAS131.
    J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear
      Models via Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010,
      doi: 10.18637/jss.v033.i01.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). Reads options.lambda2 (diagonal ridge), options.kappa_dummy and the cd_* knobs.
        
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
    def getKktViolation(self, lambda1: typing.SupportsFloat | typing.SupportsIndex = -1.0) -> float:
        """
        KKT violation of the current state at lambda1 (< 0: the last recorded lambda).
        """
    def getLambda(self) -> list[float]:
        """
        lambda_1 at every recorded step.
        """
    def getLambda2(self) -> float:
        """
        Quadratic penalty weight lambda_2 (0 for TCDLASSO).
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
        Get the total number of variable removals along the path.
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
    def getRecordGaps(self) -> list[float]:
        """
        Duality gap at every recorded step.
        """
    def getRecordKktViolations(self) -> list[float]:
        """
        KKT violation at every recorded step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isCertifiedDegraded(self) -> bool:
        """
        True once certification became unavailable along the path (A0-w boundary) and the solver fell back to LEGACY.
        """
    def isCertifiedLocalization(self) -> bool:
        """
        True if the solver localizes crossings with certificates (options.cd_certified_localization).
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
class TCDIENET_Solver:
    """
    Terminating cyclic coordinate descent informed elastic net (T-CDIENET) solver.
    
    References
    ----------
    J. Machkour, M. Muma, and D. P. Palomar, "The Informed Elastic Net for Fast Grouped
      Variable Selection and FDR Control in Genomics Research," in Proc. IEEE 9th Int.
      Workshop Comput. Adv. Multi-Sensor Adapt. Process. (CAMSAP), 2023, pp. 466-470.
    H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
    J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate
      Optimization," Ann. Appl. Stat., vol. 1, no. 2, pp. 302-332, 2007,
      doi: 10.1214/07-AOAS131.
    J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear
      Models via Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010,
      doi: 10.18637/jss.v033.i01.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], groups: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 1]"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y with a 0-based group id per original variable (`groups`, length p); every other input comes from `options` (SolverOptions). Reads options.lambda2, options.group_penalty (MEAN or CENTERING) and the cd_* knobs.
        
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
    def getKktViolation(self, lambda1: typing.SupportsFloat | typing.SupportsIndex = -1.0) -> float:
        """
        KKT violation of the current state at lambda1 (< 0: the last recorded lambda).
        """
    def getLambda(self) -> list[float]:
        """
        lambda_1 at every recorded step.
        """
    def getLambda2(self) -> float:
        """
        Quadratic penalty weight lambda_2 (0 for TCDLASSO).
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
    def getNumRemovals(self) -> int:
        """
        Get the total number of variable removals along the path.
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
    def getRecordGaps(self) -> list[float]:
        """
        Duality gap at every recorded step.
        """
    def getRecordKktViolations(self) -> list[float]:
        """
        KKT violation at every recorded step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isCertifiedDegraded(self) -> bool:
        """
        True once certification became unavailable along the path (A0-w boundary) and the solver fell back to LEGACY.
        """
    def isCertifiedLocalization(self) -> bool:
        """
        True if the solver localizes crossings with certificates (options.cd_certified_localization).
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
class TCDLASSO_Solver:
    """
    Terminating cyclic coordinate descent LASSO (T-CDLASSO) solver.
    
    References
    ----------
    R. Tibshirani, "Regression Shrinkage and Selection via the Lasso," J. R. Stat. Soc. Ser.
      B Stat. Methodol., vol. 58, no. 1, pp. 267-288, 1996.
    J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate
      Optimization," Ann. Appl. Stat., vol. 1, no. 2, pp. 302-332, 2007,
      doi: 10.1214/07-AOAS131.
    J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear
      Models via Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010,
      doi: 10.18637/jss.v033.i01.
    W. J. Fu, "Penalized Regression: The Bridge Versus the Lasso," J. Comput. Graph.
      Statist., vol. 7, pp. 397-416, 1998.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y; every other input comes from `options` (SolverOptions). Exact penalized-lasso minimizers per dummy crossing; reads the options.cd_* knobs.
        
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
    def getKktViolation(self, lambda1: typing.SupportsFloat | typing.SupportsIndex = -1.0) -> float:
        """
        KKT violation of the current state at lambda1 (< 0: the last recorded lambda).
        """
    def getLambda(self) -> list[float]:
        """
        lambda_1 at every recorded step.
        """
    def getLambda2(self) -> float:
        """
        Quadratic penalty weight lambda_2 (0 for TCDLASSO).
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
        Get the total number of variable removals along the path.
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
    def getRecordGaps(self) -> list[float]:
        """
        Duality gap at every recorded step.
        """
    def getRecordKktViolations(self) -> list[float]:
        """
        KKT violation at every recorded step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isCertifiedDegraded(self) -> bool:
        """
        True once certification became unavailable along the path (A0-w boundary) and the solver fell back to LEGACY.
        """
    def isCertifiedLocalization(self) -> bool:
        """
        True if the solver localizes crossings with certificates (options.cd_certified_localization).
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
class TCDTIKNET_Solver:
    """
    Terminating cyclic coordinate descent Tikhonov net (T-CDTIKNET) solver.
    
    References
    ----------
    A. N. Tikhonov, "Solution of Incorrectly Formulated Problems and the Regularization
      Method," Soviet Math. Dokl., vol. 4, pp. 1035-1038, 1963 (English translation of Dokl.
      Akad. Nauk SSSR 151 (1963), no. 3, 501-504).
    Y. Park, L. Reichel, G. Rodriguez, and X. Yu, "Parameter Determination for Tikhonov
      Regularization Problems in General Form," J. Comput. Appl. Math., vol. 343, pp. 12-25,
      2018, doi: 10.1016/j.cam.2018.04.049.
    J. Friedman, T. Hastie, H. Höfling, and R. Tibshirani, "Pathwise Coordinate
      Optimization," Ann. Appl. Stat., vol. 1, no. 2, pp. 302-332, 2007,
      doi: 10.1214/07-AOAS131.
    J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear
      Models via Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010,
      doi: 10.18637/jss.v033.i01.
    J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector:
      Fast High-Dimensional Variable Selection with False Discovery Rate Control," Signal
      Process., vol. 231, Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
    """
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], tikhonov_K: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], options: trexselector._core.tsolvers.ad.SolverOptions = ...) -> None:
        """
        Initializes the solver via zero-copy views of X, D and y with the p x p PSD Tikhonov matrix K (a dense array; converted to sparse internally); every other input comes from `options` (SolverOptions). Reads options.lambda2, options.kappa_dummy, options.fold_dummies and the cd_* knobs.
        
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
    def getKktViolation(self, lambda1: typing.SupportsFloat | typing.SupportsIndex = -1.0) -> float:
        """
        KKT violation of the current state at lambda1 (< 0: the last recorded lambda).
        """
    def getLambda(self) -> list[float]:
        """
        lambda_1 at every recorded step.
        """
    def getLambda2(self) -> float:
        """
        Quadratic penalty weight lambda_2 (0 for TCDLASSO).
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
        Get the total number of variable removals along the path.
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
    def getRecordGaps(self) -> list[float]:
        """
        Duality gap at every recorded step.
        """
    def getRecordKktViolations(self) -> list[float]:
        """
        KKT violation at every recorded step.
        """
    def getResiduals(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get the residual vector at the current step.
        """
    def getWarnings(self) -> list[str]:
        """
        Get all recorded warnings (dropped columns, collinear rejections, ...).
        """
    def isCertifiedDegraded(self) -> bool:
        """
        True once certification became unavailable along the path (A0-w boundary) and the solver fell back to LEGACY.
        """
    def isCertifiedLocalization(self) -> bool:
        """
        True if the solver localizes crossings with certificates (options.cd_certified_localization).
        """
    def isConnected(self) -> bool:
        """
        Check whether the solver has mapped access to data predictors.
        """
    def isFoldedDummyCoupling(self) -> bool:
        """
        True under the FOLDED dummy coupling (options.fold_dummies).
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
