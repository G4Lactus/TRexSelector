"""
Machine learning methods
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
from . import clustering
__all__: list[str] = ['ElasticNet', 'ElasticNetCV', 'L1', 'L2', 'LpNormScaler', 'NormType', 'PCA', 'PCAResult', 'RidgeCV', 'RidgeSolver', 'SVDResult', 'SVDSolver', 'ZScoreScaler', 'clustering']
class ElasticNet:
    def __init__(self) -> None:
        ...
    def coef(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Coefficient path in original predictor scale (p x n_lambda).
        """
    def converged(self) -> bool:
        """
        True if every lambda reached the convergence tolerance.
        """
    def dev_ratio(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Fraction of null deviance explained per fitted lambda (glmnet %Dev).
        """
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], alpha: typing.SupportsFloat | typing.SupportsIndex = 1.0, n_lambda: typing.SupportsInt | typing.SupportsIndex = 100, lambda_min_ratio: typing.SupportsFloat | typing.SupportsIndex = -1.0, standardize: bool = True, intercept: bool = True, use_strong_rule: bool = False, max_iter: typing.SupportsInt | typing.SupportsIndex = 100000, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-07) -> ElasticNet:
        """
        Fit the EN path over an auto-generated glmnet-style lambda grid (alpha: 1 = lasso, 0 = ridge).
        """
    def fit_grid(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], lambda_grid: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], alpha: typing.SupportsFloat | typing.SupportsIndex = 1.0, standardize: bool = True, intercept: bool = True, use_strong_rule: bool = False, max_iter: typing.SupportsInt | typing.SupportsIndex = 100000, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-07) -> ElasticNet:
        """
        Fit the EN path at an explicit lambda grid (sorted descending internally); use to evaluate at exactly glmnet's lambda sequence.
        """
    def intercepts(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Intercept per lambda (zero if intercept=False).
        """
    def lambdas(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Lambda grid used (descending).
        """
    def n_nonconverged(self) -> int:
        """
        Number of lambda points that hit max_iter without converging.
        """
    def predict(self, X_new: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Predictions per lambda: (X_new @ coef) + intercept.
        """
class ElasticNetCV:
    def __init__(self) -> None:
        ...
    def cv_1se(self) -> float:
        """
        Largest lambda within 1 SE of the min.
        """
    def cv_min(self) -> float:
        """
        lambda minimizing mean CV MSE.
        """
    def cv_mse(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Mean CV MSE per lambda.
        """
    def cv_sem(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Standard error of the CV MSE per lambda.
        """
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], alpha: typing.SupportsFloat | typing.SupportsIndex = 0.0, n_folds: typing.SupportsInt | typing.SupportsIndex = 10, n_lambda: typing.SupportsInt | typing.SupportsIndex = 100, lambda_min_ratio: typing.SupportsFloat | typing.SupportsIndex = -1.0, seed: typing.SupportsInt | typing.SupportsIndex = 0, standardize: bool = True, intercept: bool = True, max_iter: typing.SupportsInt | typing.SupportsIndex = 100000, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-07) -> ElasticNetCV:
        """
        Fit K-fold CV over a glmnet-style EN grid (alpha: 1 = lasso, 0 = ridge).
        """
    def index_1se(self) -> int:
        """
        0-based index of lambda.1se.
        """
    def index_min(self) -> int:
        """
        0-based index of lambda.min.
        """
    def lambdas(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Lambda grid used (descending).
        """
class LpNormScaler:
    def __init__(self, norm_type: NormType = ..., center: bool = True, scale: bool = True) -> None:
        ...
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], threshold: typing.SupportsFloat | typing.SupportsIndex = 1e-12) -> LpNormScaler:
        ...
    def fit_transform(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], threshold: typing.SupportsFloat | typing.SupportsIndex = 1e-12) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Fit on X, then return a normalized copy (any memory order; X unchanged).
        """
    def fit_transform_inplace(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], threshold: typing.SupportsFloat | typing.SupportsIndex = 1e-12) -> LpNormScaler:
        ...
    def get_center(self) -> bool:
        ...
    def get_centers(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def get_dropped_indices(self) -> list[int]:
        ...
    def get_norm_type(self) -> NormType:
        ...
    def get_scale(self) -> bool:
        ...
    def get_scales(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def inverse_transform(self, X_normed: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Return a denormalized copy of X_normed (any memory order; input unchanged).
        """
    def inverse_transform_inplace(self, X_normed: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        ...
    def is_fitted(self) -> bool:
        ...
    def load(self, filename: str) -> None:
        ...
    def save(self, filename: str) -> None:
        ...
    def transform(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Return a normalized copy of X (any memory order accepted; X unchanged).
        """
    def transform_inplace(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        ...
class NormType:
    """
    Members:
    
      L1
    
      L2
    """
    L1: typing.ClassVar[NormType]  # value = <NormType.L1: 0>
    L2: typing.ClassVar[NormType]  # value = <NormType.L2: 1>
    __members__: typing.ClassVar[dict[str, NormType]]  # value = {'L1': <NormType.L1: 0>, 'L2': <NormType.L2: 1>}
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
class PCA:
    def __init__(self, center: bool = True, normalize: bool = True) -> None:
        ...
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], M: typing.SupportsInt | typing.SupportsIndex) -> PCAResult:
        ...
    def get_explained_variance(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def get_loadings(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    def get_mean(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[1, n]"]:
        ...
    def get_means(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[1, n]"]:
        ...
    def get_norms(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[1, n]"]:
        ...
    def restore(self) -> None:
        ...
    def transform(self, X_new: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
class PCAResult:
    def __init__(self) -> None:
        ...
    @property
    def V(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    @V.setter
    def V(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> None:
        ...
    @property
    def Z(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    @Z.setter
    def Z(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> None:
        ...
    @property
    def explained_variance(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    @explained_variance.setter
    def explained_variance(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
class RidgeCV:
    def __init__(self) -> None:
        ...
    def cv_1se(self) -> float:
        """
        Largest lambda within 1 SE of the min.
        """
    def cv_min(self) -> float:
        """
        lambda minimizing mean CV MSE.
        """
    def cv_mse(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Mean CV MSE per lambda.
        """
    def cv_sem(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Standard error of the CV MSE per lambda.
        """
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], n_folds: typing.SupportsInt | typing.SupportsIndex = 10, n_lambda: typing.SupportsInt | typing.SupportsIndex = 1000, lambda_ratio: typing.SupportsFloat | typing.SupportsIndex = 1000.0, seed: typing.SupportsInt | typing.SupportsIndex = 0) -> RidgeCV:
        """
        Fit K-fold CV over a log-spaced ridge grid.
        """
    def index_1se(self) -> int:
        """
        0-based index of lambda.1se.
        """
    def index_min(self) -> int:
        """
        0-based index of lambda.min.
        """
    def lambdas(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Lambda grid used (descending).
        """
class RidgeSolver:
    @staticmethod
    def solve(X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], lambda_val: typing.SupportsFloat | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def __init__(self) -> None:
        ...
class SVDResult:
    def __init__(self) -> None:
        ...
    @property
    def S(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    @S.setter
    def S(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]) -> None:
        ...
    @property
    def U(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    @U.setter
    def U(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> None:
        ...
    @property
    def V(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    @V.setter
    def V(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> None:
        ...
class SVDSolver:
    def __init__(self) -> None:
        ...
    def compute(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], M: typing.SupportsInt | typing.SupportsIndex) -> SVDResult:
        ...
class ZScoreScaler:
    def __init__(self, center: bool = True, scale: bool = True) -> None:
        ...
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], threshold: typing.SupportsFloat | typing.SupportsIndex = 1e-12) -> ZScoreScaler:
        ...
    def fit_transform(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], threshold: typing.SupportsFloat | typing.SupportsIndex = 1e-12) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Fit on X, then return a scaled copy (any memory order; X unchanged).
        """
    def fit_transform_inplace(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"], threshold: typing.SupportsFloat | typing.SupportsIndex = 1e-12) -> ZScoreScaler:
        ...
    def get_center(self) -> bool:
        ...
    def get_centers(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def get_dropped_indices(self) -> list[int]:
        ...
    def get_scale(self) -> bool:
        ...
    def get_scales(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def inverse_transform(self, X_scaled: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Return an unscaled copy of X_scaled (any memory order; input unchanged).
        """
    def inverse_transform_inplace(self, X_scaled: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        ...
    def is_fitted(self) -> bool:
        ...
    def load(self, filename: str) -> None:
        ...
    def save(self, filename: str) -> None:
        ...
    def transform(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Return a scaled copy of X (any memory order accepted; X unchanged).
        """
    def transform_inplace(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        ...
L1: NormType  # value = <NormType.L1: 0>
L2: NormType  # value = <NormType.L2: 1>
