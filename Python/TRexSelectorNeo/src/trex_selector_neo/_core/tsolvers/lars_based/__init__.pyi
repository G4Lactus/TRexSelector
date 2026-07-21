"""
LARS-based T-Solvers
"""
from __future__ import annotations
import numpy
import numpy.typing
import trex_selector_neo._core.tsolvers
import typing
__all__: list[str] = ['TENETAug_Solver', 'TENET_Solver', 'TIENETAug_Solver', 'TLARS_Solver', 'TLASSO_Solver', 'TSTAGEWISE_Solver', 'TSTEPWISE_Solver']
class TENETAug_Solver:
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], lambda2: typing.SupportsFloat | typing.SupportsIndex, normalize: bool = True, intercept: bool = True, verbose: bool = False, scaling_mode: trex_selector_neo._core.tsolvers.ScalingMode = ..., use_lars_inner: bool = False) -> None:
        """
        Initializes the augmented-LASSO Elastic Net solver with an explicit secondary L2 penalty `lambda2`. Set `use_lars_inner` to run the inner path with a pure-LARS solver (never drops variables).
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
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumSteps(self) -> int:
        """
        Quantify the number of path steps successfully taken thus far.
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
        Restore solver state parameters based on fresh target views.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
        """
    def setExchangeableTie(self, alpha: typing.SupportsFloat | typing.SupportsIndex, floor: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> None:
        """
        Configure exchangeability-calibrated stochastic tie-breaking for greedy candidate selection (TOMP/TAFS); alpha <= 0 turns it off.
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
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], lambda2: typing.SupportsFloat | typing.SupportsIndex, normalize: bool = True, intercept: bool = True, verbose: bool = False, scaling_mode: trex_selector_neo._core.tsolvers.ScalingMode = ...) -> None:
        """
        Initializes the TENET solver array, using an explicit secondary L2 penalty `lambda2`.
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
        Restore solver state parameters based on fresh target views.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
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
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], lambda2: typing.SupportsFloat | typing.SupportsIndex, groups: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 1]"], normalize: bool = True, intercept: bool = True, verbose: bool = False, scaling_mode: trex_selector_neo._core.tsolvers.ScalingMode = ..., use_lars_inner: bool = False) -> None:
        """
        Initializes the augmented-LASSO Informed Elastic Net solver with a group-ridge penalty `lambda2` and a 0-based group id per original variable (`groups`, length p).
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
        Restore solver state parameters based on fresh target views.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
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
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], normalize: bool = True, intercept: bool = True, verbose: bool = False, scaling_mode: trex_selector_neo._core.tsolvers.ScalingMode = ...) -> None:
        """
        Initializes the solver via zero-copy views to facilitate safe in-place mutations.
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
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumSteps(self) -> int:
        """
        Quantify the number of path steps successfully taken thus far.
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
        Restore solver state parameters based on fresh target views.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
        """
    def setKahanRefreshInterval(self, interval: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the Kahan summation refresh interval (call before executeStep).
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
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], normalize: bool = True, intercept: bool = True, verbose: bool = False, scaling_mode: trex_selector_neo._core.tsolvers.ScalingMode = ...) -> None:
        """
        Initializes the solver via zero-copy views to facilitate safe in-place mutations.
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
        Restore solver state parameters based on fresh target views.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
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
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], normalize: bool = True, intercept: bool = True, verbose: bool = False, scaling_mode: trex_selector_neo._core.tsolvers.ScalingMode = ...) -> None:
        """
        Initializes the solver via zero-copy views to facilitate safe in-place mutations.
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
        Restore solver state parameters based on fresh target views.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
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
    def __init__(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], D: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], normalize: bool = True, intercept: bool = True, verbose: bool = False, scaling_mode: trex_selector_neo._core.tsolvers.ScalingMode = ...) -> None:
        """
        Initializes the solver via zero-copy views to facilitate safe in-place mutations.
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
    def getNumActives(self) -> int:
        """
        Calculate how many variables are currently active.
        """
    def getNumSteps(self) -> int:
        """
        Quantify the number of path steps successfully taken thus far.
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
        Restore solver state parameters based on fresh target views.
        """
    def save(self, filepath: str) -> None:
        """
        Serialize calculation state directly to disk utilizing Cereal.
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
