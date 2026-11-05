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
    """
    Elastic-net coordinate-descent path (glmnet-equivalent).
    
    References
    ----------
    H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
    J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear
      Models via Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010,
      doi: 10.18637/jss.v033.i01.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, low_memory: bool, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        low_memory=False (default): the coordinate descent runs on a centered and standardized working copy of X (one n x p matrix while fitting). low_memory=True: it runs on X itself through the caller's view (a MemoryMappedMatrix.to_numpy() view included) with no allocation of the size of X; same estimator and grid, the two engines agree to rounding (about 1e-12 in the coefficients). n_threads: the thread budget of the fit's matrix products (None: the library's default budget, utils.get_num_threads(), initially 1).
        """
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
    def low_memory(self) -> bool:
        """
        Whether the raw-X (no working copy) engine is selected.
        """
    def n_nonconverged(self) -> int:
        """
        Number of lambda points that hit max_iter without converging.
        """
    def num_threads(self) -> int:
        """
        The thread budget of the fit's matrix products.
        """
    def predict(self, X_new: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Predictions per lambda: (X_new @ coef) + intercept.
        """
    def set_low_memory(self, low_memory: bool) -> None:
        """
        Switch the engine; takes effect at the next fit().
        """
    def set_num_threads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the fit's matrix products (>= 1).
        """
class ElasticNetCV:
    """
    K-fold cross-validated elastic net (coordinate-descent per fold).
    
    References
    ----------
    H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 67, no. 2, pp. 301-320, 2005.
    J. Friedman, T. Hastie, and R. Tibshirani, "Regularization Paths for Generalized Linear
      Models via Coordinate Descent," J. Stat. Softw., vol. 33, no. 1, pp. 1-22, 2010,
      doi: 10.18637/jss.v033.i01.
    R. Tibshirani, J. Bien, J. Friedman, T. Hastie, N. Simon, J. Taylor, and R. J.
      Tibshirani, "Strong Rules for Discarding Predictors in Lasso-Type Problems," J. R.
      Stat. Soc. Ser. B Stat. Methodol., vol. 74, no. 2, pp. 245-266, 2012,
      doi: 10.1111/j.1467-9868.2011.01004.x.
    L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, Classification and Regression
      Trees. Wadsworth, Belmont, CA, 1984.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, low_memory: bool, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        low_memory=True runs the full-data fit and every fold on X itself with a 0/1 training mask (no allocation of the size of X); see ElasticNet. n_threads: the thread budget of the matrix products of the fit and its folds (None: the library's default budget, utils.get_num_threads(), initially 1).
        """
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
    def grid_truncated(self) -> bool:
        """
        Whether the deviance rules stopped the path before the requested n_lambda.
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
    def low_memory(self) -> bool:
        """
        Whether the raw-X engine is selected.
        """
    def num_threads(self) -> int:
        """
        The thread budget of the matrix products of the fit and its folds.
        """
    def selection_at_grid_edge(self) -> bool:
        """
        Whether lambda.min or lambda.1se sits on the first or last grid point.
        """
    def set_low_memory(self, low_memory: bool) -> None:
        """
        Switch the engine; takes effect at the next fit().
        """
    def set_num_threads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the matrix products of the fit and its folds (>= 1).
        """
class LpNormScaler:
    def __init__(self, norm_type: NormType = ..., center: bool = True, scale: bool = True, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        n_threads: the thread budget of the column passes (None: the library's default budget, utils.get_num_threads(), initially 1).
        """
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], threshold: typing.SupportsFloat | typing.SupportsIndex = 1e-12) -> LpNormScaler:
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
    def get_num_threads(self) -> int:
        """
        The thread budget of the column passes.
        """
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
    def set_num_threads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the column passes (>= 1).
        """
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
    @typing.overload
    def __eq__(self, other: NormType) -> bool:
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
    def __ne__(self, other: NormType) -> bool:
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
class PCA:
    """
    Fits PCA to a data matrix with optional in-place centering (RAII).
    
    References
    ----------
    G. Guennebaud, B. Jacob et al., Eigen v3, https://libeigen.gitlab.io/, 2010.
    """
    def __init__(self, center: bool = True, normalize: bool = True, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        n_threads: the thread budget of the matrix products of fit() and transform() (None: the library's default budget, utils.get_num_threads(), initially 1).
        """
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
    def get_num_threads(self) -> int:
        """
        The thread budget of the matrix products.
        """
    def restore(self) -> None:
        ...
    def set_num_threads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the next fit() (>= 1).
        """
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
    """
    K-fold cross-validated ridge regression (SVD path per fold).
    
    References
    ----------
    A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal
      Problems," Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article
      in the Special 40th Anniversary Issue).
    G. Guennebaud, B. Jacob et al., Eigen v3, https://libeigen.gitlab.io/, 2010.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, low_memory: bool, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        low_memory=True: the dual (kernel) fold engine — n x n kernels accumulated from column blocks of X read in place, no p-dimensional coefficients; same estimator, grid and rank rule, agreement to numerical precision; memory O(n^2) instead of O(n p), the route for p >> n. n_threads: the thread budget of the fit's matrix products (None: the library's default budget, utils.get_num_threads(), initially 1).
        """
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
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], n_folds: typing.SupportsInt | typing.SupportsIndex = 10, n_lambda: typing.SupportsInt | typing.SupportsIndex = 100, lambda_ratio: typing.SupportsFloat | typing.SupportsIndex = 1000.0, seed: typing.SupportsInt | typing.SupportsIndex = 0) -> RidgeCV:
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
    def interpolates(self) -> bool:
        """
        Whether the smallest training fold has p >= n_train, so the lambda -> 0 end of the grid interpolates.
        """
    def lambdas(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Lambda grid used (descending).
        """
    def low_memory(self) -> bool:
        """
        Whether the kernel fold engine is selected.
        """
    def num_threads(self) -> int:
        """
        The thread budget of the fit's matrix products.
        """
    def selection_at_grid_edge(self) -> bool:
        """
        Whether lambda.min or lambda.1se sits on the first or last grid point.
        """
    def set_low_memory(self, low_memory: bool) -> None:
        """
        Switch the fold engine; takes effect at the next fit().
        """
    def set_num_threads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the fit's matrix products (>= 1).
        """
class RidgeSolver:
    """
    Solves ridge regression for a single lambda value.
    
    References
    ----------
    A. E. Hoerl and R. W. Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal
      Problems," Technometrics, vol. 42, no. 1, pp. 80-86, 2000 (Reprint of the 1970 article
      in the Special 40th Anniversary Issue).
    """
    @staticmethod
    def solve(X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], lambda_val: typing.SupportsFloat | typing.SupportsIndex, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        n_threads: the thread budget of the matrix products (None: the library's default budget, utils.get_num_threads(), initially 1).
        """
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
    """
    Computes the top-M truncated SVD of a matrix.
    
    References
    ----------
    G. Guennebaud, B. Jacob et al., Eigen v3, https://libeigen.gitlab.io/, 2010.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, min_dim_thresh: typing.SupportsInt | typing.SupportsIndex = 10000, block_size: typing.SupportsInt | typing.SupportsIndex = 1024, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        n_threads: the thread budget of the matrix products (None: the library's default budget, utils.get_num_threads(), initially 1). Dispatch thresholds: the direct Jacobi path copies X and computes every singular triplet; when min(n, p) exceeds min_dim_thresh and M < min(n, p) / 10 the randomized path runs two block-wise passes over X (no copy; the route for a memory-mapped X), and p > 2n always takes the Gram path. Lowering min_dim_thresh selects the randomized path for smaller designs; block_size is the column block of the block-wise passes.
        """
    def compute(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], M: typing.SupportsInt | typing.SupportsIndex) -> SVDResult:
        ...
    def num_threads(self) -> int:
        """
        The thread budget of the matrix products.
        """
    def set_num_threads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the matrix products (>= 1).
        """
class ZScoreScaler:
    def __init__(self, center: bool = True, scale: bool = True, n_threads: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        n_threads: the thread budget of the column passes (None: the library's default budget, utils.get_num_threads(), initially 1).
        """
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"], threshold: typing.SupportsFloat | typing.SupportsIndex = 1e-12) -> ZScoreScaler:
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
    def get_num_threads(self) -> int:
        """
        The thread budget of the column passes.
        """
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
    def set_num_threads(self, n_threads: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the thread budget of the column passes (>= 1).
        """
    def transform(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.f_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Return a scaled copy of X (any memory order accepted; X unchanged).
        """
    def transform_inplace(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.writeable", "flags.f_contiguous"]) -> None:
        ...
L1: NormType  # value = <NormType.L1: 0>
L2: NormType  # value = <NormType.L2: 1>
