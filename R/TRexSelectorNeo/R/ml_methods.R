#' @title Z-Score Scaler
#'
#' @description Standardizes features by removing the mean and scaling to unit
#'   variance (Bessel-corrected, n - 1). The `center`/`scale` arguments follow
#'   R's \code{\link[base]{scale}()}: with \code{center = FALSE, scale = TRUE}
#'   columns are divided by the root-mean-square around 0.
#'
#' @details The \code{*_inplace} methods mutate \code{X} directly (zero-copy)
#'   when it is a double-storage matrix; other inputs are coerced to a double
#'   matrix first, in which case the coerced copy is modified. The (modified)
#'   matrix is returned in both cases.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(50 * 5), 50, 5)
#' scaler <- ZScoreScaler$new()
#' scaler$fit(X)
#' X_std <- scaler$transform_inplace(X)
#' scaler$get_centers()
#'
#' @export
ZScoreScaler <- R6::R6Class("ZScoreScaler",
  private = list(
    cpp_ptr = NULL
  ),
  public = list(

    #' @description Initialize the ZScoreScaler
    #'
    #' @param center Logical, whether to center columns to mean 0 (default: TRUE)
    #' @param scale Logical, whether to scale columns to unit SD (default: TRUE);
    #'   around 0 instead of the mean when \code{center = FALSE}, as in R's scale().
    initialize = function(center = TRUE, scale = TRUE) {
      private$cpp_ptr <- zscore_scaler_create(center, scale)
    },

    #' @description Compute the centers and scales to be used for later scaling.
    #'
    #' @param X A numeric matrix (n x p).
    #' @param threshold Double, lower bound for the scale statistic (default: 1e-12).
    #'
    #' @return Value.
    fit = function(X, threshold = 1e-12) {
      X <- .as_double_matrix(X)
      zscore_scaler_fit(private$cpp_ptr, X, threshold)
      invisible(self)
    },

    #' @description Perform standardization by centering and scaling in-place.
    #'
    #' @param X A numeric matrix (n x p).
    #'
    #' @return The modified matrix `X`.
    transform_inplace = function(X) {
      X <- .as_double_matrix(X)
      zscore_scaler_transform_inplace(private$cpp_ptr, X)
      return(X)
    },

    #' @description Inverse transform to original scale.
    #'
    #' @param X A numeric matrix (n x p).
    #'
    #' @return The modified matrix `X`.
    inverse_transform_inplace = function(X) {
      X <- .as_double_matrix(X)
      zscore_scaler_inverse_transform_inplace(private$cpp_ptr, X)
      return(X)
    },

    #' @description Fit and transform in one call, like R's scale().
    #'
    #' @param X A numeric matrix (n x p).
    #' @param threshold Double, lower bound for the scale statistic (default: 1e-12).
    #'
    #' @return The modified matrix `X`.
    fit_transform_inplace = function(X, threshold = 1e-12) {
      X <- .as_double_matrix(X)
      zscore_scaler_fit_transform_inplace(private$cpp_ptr, X, threshold)
      return(X)
    },

    #' @description Check if scaler is fitted.
    #'
    #' @return Value.
    is_fitted = function() {
      zscore_scaler_is_fitted(private$cpp_ptr)
    },

    #' @description Get dropped (degenerate) column indices. These columns get
    #'   scale factor 1 and are still centered (sklearn semantics).
    #'
    #' @return Value.
    get_dropped_indices = function() {
      # Convert 0-based Cpp column indices to 1-based R convention
      zscore_scaler_get_dropped_indices(private$cpp_ptr) + 1L
    },

    #' @description Check if scaler centers data.
    #'
    #' @return Value.
    get_center = function() {
      zscore_scaler_get_center(private$cpp_ptr)
    },

    #' @description Check if scaler scales data to unit variance.
    #'
    #' @return Value.
    get_scale = function() {
      zscore_scaler_get_scale(private$cpp_ptr)
    },

    #' @description Get the learned centers (column means).
    #'
    #' @return Value.
    get_centers = function() {
      zscore_scaler_get_centers(private$cpp_ptr)
    },

    #' @description Get the learned scales (standard deviations).
    #'
    #' @return Value.
    get_scales = function() {
      zscore_scaler_get_scales(private$cpp_ptr)
    },

    #' @description Save model state to file.
    #'
    #' @param filename String path to save output.
    #'
    #' @return Value.
    save = function(filename) {
      if (!is.character(filename) || length(filename) != 1) {
        stop("filename must be a single string.")
      }
      zscore_scaler_save(private$cpp_ptr, filename)
      invisible(self)
    },

    #' @description Load model state from file.
    #'
    #' @param filename String path to load input from.
    #'
    #' @return Value.
    load = function(filename) {
      if (!is.character(filename) || length(filename) != 1) {
        stop("filename must be a single string.")
      }
      zscore_scaler_load(private$cpp_ptr, filename)
      invisible(self)
    }
  )
)


#' @title Lp-Norm Scaler
#'
#' @description Scales input features to unit norm based on the chosen L1 or
#'   L2 formulation. The `center`/`scale` arguments follow R's
#'   \code{\link[base]{scale}()}: the norm is computed around the applied
#'   center (around 0 when \code{center = FALSE}).
#'
#' @details The \code{*_inplace} methods mutate \code{X} directly (zero-copy)
#'   when it is a double-storage matrix; other inputs are coerced to a double
#'   matrix first, in which case the coerced copy is modified. The (modified)
#'   matrix is returned in both cases.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(50 * 5), 50, 5)
#' scaler <- LpNormScaler$new(norm_type = 2)
#' scaler$fit(X)
#' X_norm <- scaler$transform_inplace(X)
#'
#' @export
LpNormScaler <- R6::R6Class("LpNormScaler",
  private = list(
    cpp_ptr = NULL
  ),
  public = list(

    #' @description Initialize the LpNormScaler
    #'
    #' @param norm_type Integer, 1 for L1 norm, 2 for L2 norm (default: 2)
    #' @param center Logical, whether to center columns to mean 0 (default: TRUE)
    #' @param scale Logical, whether to scale columns to unit Lp norm (default: TRUE)
    initialize = function(norm_type = 2, center = TRUE, scale = TRUE) {
      if (!(norm_type %in% c(1, 2))) {
        stop("norm_type must be 1 (L1) or 2 (L2)")
      }
      private$cpp_ptr <- lpnorm_scaler_create(norm_type, center, scale)
    },

    #' @description Compute the norm parameters.
    #'
    #' @param X A numeric matrix (n x p).
    #' @param threshold Double, lower bound for the scale statistic (default: 1e-12).
    #'
    #' @return Value.
    fit = function(X, threshold = 1e-12) {
      X <- .as_double_matrix(X)
      lpnorm_scaler_fit(private$cpp_ptr, X, threshold)
      invisible(self)
    },

    #' @description Perform standardization in-place.
    #'
    #' @param X A numeric matrix (n x p).
    #'
    #' @return The modified matrix `X`.
    transform_inplace = function(X) {
      X <- .as_double_matrix(X)
      lpnorm_scaler_transform_inplace(private$cpp_ptr, X)
      return(X)
    },

    #' @description Inverse transform to original scale.
    #'
    #' @param X A numeric matrix (n x p).
    #'
    #' @return The modified matrix `X`.
    inverse_transform_inplace = function(X) {
      X <- .as_double_matrix(X)
      lpnorm_scaler_inverse_transform_inplace(private$cpp_ptr, X)
      return(X)
    },

    #' @description Fit and transform in one call, like R's scale().
    #'
    #' @param X A numeric matrix (n x p).
    #' @param threshold Double, lower bound for the scale statistic (default: 1e-12).
    #'
    #' @return The modified matrix `X`.
    fit_transform_inplace = function(X, threshold = 1e-12) {
      X <- .as_double_matrix(X)
      lpnorm_scaler_fit_transform_inplace(private$cpp_ptr, X, threshold)
      return(X)
    },

    #' @description Check if scaler is fitted.
    #'
    #' @return Value.
    is_fitted = function() {
      lpnorm_scaler_is_fitted(private$cpp_ptr)
    },

    #' @description Get dropped (degenerate) column indices. These columns get
    #'   scale factor 1 and are still centered (sklearn semantics).
    #'
    #' @return Value.
    get_dropped_indices = function() {
      # Convert 0-based Cpp column indices to 1-based R convention
      lpnorm_scaler_get_dropped_indices(private$cpp_ptr) + 1L
    },

    #' @description Get the learned centers (column means).
    #'
    #' @return Value.
    get_centers = function() {
      lpnorm_scaler_get_centers(private$cpp_ptr)
    },

    #' @description Get the learned scales.
    #'
    #' @return Value.
    get_scales = function() {
      lpnorm_scaler_get_scales(private$cpp_ptr)
    },

    #' @description Check if scaler centers data.
    #'
    #' @return Value.
    get_center = function() {
      lpnorm_scaler_get_center(private$cpp_ptr)
    },

    #' @description Check if scaler scales data to unit norm.
    #'
    #' @return Value.
    get_scale = function() {
      lpnorm_scaler_get_scale(private$cpp_ptr)
    },

    #' @description Get norm type (1 for L1, 2 for L2).
    #'
    #' @return Value.
    get_norm_type = function() {
      lpnorm_scaler_get_norm_type(private$cpp_ptr)
    },

    #' @description Save model state to file.
    #'
    #' @param filename String path to save output.
    #'
    #' @return Value.
    save = function(filename) {
      if (!is.character(filename) || length(filename) != 1) {
        stop("filename must be a single string.")
      }
      lpnorm_scaler_save(private$cpp_ptr, filename)
      invisible(self)
    },

    #' @description Load model state from file.
    #'
    #' @param filename String path to load input from.
    #'
    #' @return Value.
    load = function(filename) {
      if (!is.character(filename) || length(filename) != 1) {
        stop("filename must be a single string.")
      }
      lpnorm_scaler_load(private$cpp_ptr, filename)
      invisible(self)
    }
  )
)



#' @title Ridge Regression with Cross-Validation
#'
#' @description Performs explicit K-Fold cross-validation for Ridge Regression model selection.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(50 * 5), 50, 5)
#' y <- X[, 1] * 2 + rnorm(50)
#' cv <- RidgeCV$new()
#' cv$fit(X, y, num_folds = 5)
#' cv$cv_min()
#'
#' @export
RidgeCV <- R6::R6Class("RidgeCV",
  private = list(
    cpp_ptr = NULL
  ),
  public = list(
    #' @description Initialize the RidgeCV object.
    initialize = function() {
      private$cpp_ptr <- ridge_cv_create()
    },

    #' @description Fit the CV model.
    #'
    #' @param X Predictor matrix.
    #' @param y Response vector.
    #' @param num_folds Number of folds (default 10).
    #' @param n_lambda Number of lambdas in path (default 100).
    #' @param lambda_ratio Ratio of max to min lambda (default 1000).
    #' @param seed Random seed for CV splits (default 0).
    #'
    #' @return Value.
    fit = function(X, y, num_folds = 10, n_lambda = 100, lambda_ratio = 1000.0, seed = 0) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(y), nrow(X) == length(y))
      ridge_cv_fit(private$cpp_ptr, X, as.numeric(y), as.integer(num_folds), as.integer(n_lambda),
                   as.numeric(lambda_ratio), as.integer(seed))

      invisible(self)
    },

    #' @description Get lambda that minimizes CV error.
    #'
    #' @return Value.
    cv_min = function() {
      ridge_cv_min(private$cpp_ptr)
    },

    #' @description Get 1SE rule lambda.
    #'
    #' @return Value.
    cv_1se = function() {
      ridge_cv_1se(private$cpp_ptr)
    },

    #' @description Get index of minimum CV error.
    #'
    #' @return Value.
    index_min = function() {
      # Add 1 since R is 1-indexed
      ridge_cv_index_min(private$cpp_ptr) + 1L
    },

    #' @description Get index of 1SE rule lambda.
    #'
    #' @return Value.
    index_1se = function() {
      # Add 1 since R is 1-indexed
      ridge_cv_index_1se(private$cpp_ptr) + 1L
    },

    #' @description Get the sequence of lambdas used.
    #'
    #' @return Value.
    get_lambdas = function() {
      ridge_cv_get_lambdas(private$cpp_ptr)
    },

    #' @description Get the CV mean squared errors for each lambda.
    #'
    #' @return Value.
    get_cv_errors = function() {
      ridge_cv_get_cv_errors(private$cpp_ptr)
    },

    #' @description Get the standard error of the CV errors.
    #'
    #' @return Value.
    get_cv_std = function() {
      ridge_cv_get_cv_std(private$cpp_ptr)
    }
  )
)


# ==============================================================================
# ElasticNet
# ==============================================================================

#' @title Elastic Net (coordinate-descent path)
#'
#' @description Fits the elastic-net coefficient path via coordinate descent
#'   over a glmnet-style lambda grid (or an explicit grid). Reports coefficients
#'   in the original predictor scale, intercepts, the deviance-ratio path, and
#'   predictions for new data. The mixing parameter \code{alpha} interpolates
#'   between lasso (\code{alpha = 1}) and ridge (\code{alpha = 0}).
#'
#' @examples
#' set.seed(1)
#' n <- 50; p <- 8
#' X <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + X[, 3] - X[, 5] + rnorm(n)
#' en <- ElasticNet$new()
#' en$fit(X, y, alpha = 1.0)
#' dim(en$get_coef())        # p x n_lambda
#' en$predict(X[1:5, ])
#'
#' @export
ElasticNet <- R6::R6Class("ElasticNet",
  private = list(
    cpp_ptr = NULL
  ),
  public = list(
    #' @description Initialize the ElasticNet object.
    initialize = function() {
      private$cpp_ptr <- enet_create()
    },

    #' @description Fit the EN path over an auto-generated glmnet-style grid.
    #'
    #' @param X Predictor matrix (n x p).
    #' @param y Response vector (n).
    #' @param alpha Mixing in \code{[0, 1]}: 1 = lasso, 0 = ridge (default 1).
    #' @param n_lambda Grid size (default 100).
    #' @param lambda_min_ratio Ratio lambda_min/lambda_max; negative (default)
    #'   auto-selects glmnet's rule.
    #' @param standardize Standardize columns to unit SD (default TRUE).
    #' @param intercept Fit an unpenalized intercept (default TRUE).
    #' @param use_strong_rule Enable the sequential strong rule (default FALSE).
    #' @param max_iter Max CD sweeps per lambda (default 100000).
    #' @param tol Convergence tolerance (default 1e-7).
    #' @return Invisibly returns \code{self}.
    fit = function(X, y, alpha = 1.0, n_lambda = 100, lambda_min_ratio = -1.0,
                   standardize = TRUE, intercept = TRUE, use_strong_rule = FALSE,
                   max_iter = 100000, tol = 1e-7) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(y), nrow(X) == length(y))
      enet_fit(private$cpp_ptr, X, as.numeric(y), as.numeric(alpha),
               as.integer(n_lambda), as.numeric(lambda_min_ratio),
               as.logical(standardize), as.logical(intercept),
               as.logical(use_strong_rule), as.integer(max_iter), as.numeric(tol))
      invisible(self)
    },

    #' @description Fit the EN path at an explicit lambda grid (sorted
    #'   descending internally); use to evaluate at exactly glmnet's sequence.
    #'
    #' @param X Predictor matrix (n x p).
    #' @param y Response vector (n).
    #' @param lambda_grid Explicit lambda values (any order, > 0).
    #' @param alpha Mixing in \code{[0, 1]} (default 1).
    #' @param standardize Standardize columns to unit SD (default TRUE).
    #' @param intercept Fit an unpenalized intercept (default TRUE).
    #' @param use_strong_rule Enable the sequential strong rule (default FALSE).
    #' @param max_iter Max CD sweeps per lambda (default 100000).
    #' @param tol Convergence tolerance (default 1e-7).
    #' @return Invisibly returns \code{self}.
    fit_grid = function(X, y, lambda_grid, alpha = 1.0, standardize = TRUE,
                        intercept = TRUE, use_strong_rule = FALSE,
                        max_iter = 100000, tol = 1e-7) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(y), nrow(X) == length(y), is.numeric(lambda_grid))
      enet_fit_grid(private$cpp_ptr, X, as.numeric(y), as.numeric(lambda_grid),
                    as.numeric(alpha), as.logical(standardize),
                    as.logical(intercept), as.logical(use_strong_rule),
                    as.integer(max_iter), as.numeric(tol))
      invisible(self)
    },

    #' @description Coefficient path in original predictor scale (p x n_lambda).
    #' @return Numeric matrix.
    get_coef = function() {
      enet_get_coef(private$cpp_ptr)
    },

    #' @description Intercept per lambda (zero if \code{intercept = FALSE}).
    #' @return Numeric vector.
    get_intercepts = function() {
      enet_get_intercepts(private$cpp_ptr)
    },

    #' @description Lambda grid actually used (descending).
    #' @return Numeric vector.
    get_lambdas = function() {
      enet_get_lambdas(private$cpp_ptr)
    },

    #' @description Fraction of null deviance explained per lambda (glmnet %Dev).
    #' @return Numeric vector.
    get_dev_ratio = function() {
      enet_get_dev_ratio(private$cpp_ptr)
    },

    #' @description Whether every lambda reached the convergence tolerance.
    #' @return Logical scalar.
    converged = function() {
      enet_converged(private$cpp_ptr)
    },

    #' @description Predictions per lambda for new data.
    #' @param X_new Predictor matrix with the same number of columns as the fit.
    #' @return Numeric matrix (nrow(X_new) x n_lambda).
    predict = function(X_new) {
      if (!is.matrix(X_new)) X_new <- as.matrix(X_new)
      enet_predict(private$cpp_ptr, X_new)
    }
  )
)


# ==============================================================================
# ElasticNetCV
# ==============================================================================

#' @title Elastic Net with Cross-Validation
#'
#' @description Selects the elastic-net penalty by K-fold cross-validation over
#'   a shared glmnet-style lambda grid (coordinate descent per fold), returning
#'   \code{lambda.min} and \code{lambda.1se}. The mixing parameter \code{alpha}
#'   interpolates between lasso (\code{alpha = 1}) and ridge (\code{alpha = 0}).
#'
#' @examples
#' set.seed(1)
#' n <- 60; p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 - X[, 4] + rnorm(n)
#' cv <- ElasticNetCV$new()
#' cv$fit(X, y, alpha = 0.5)
#' cv$cv_min()
#'
#' @export
ElasticNetCV <- R6::R6Class("ElasticNetCV",
  private = list(
    cpp_ptr = NULL
  ),
  public = list(
    #' @description Initialize the ElasticNetCV object.
    initialize = function() {
      private$cpp_ptr <- enet_cv_create()
    },

    #' @description Fit K-fold CV over a glmnet-style grid.
    #'
    #' @param X Predictor matrix (n x p).
    #' @param y Response vector (n).
    #' @param alpha Mixing in \code{[0, 1]}: 1 = lasso, 0 = ridge (default 0).
    #' @param n_folds Number of folds (default 10).
    #' @param n_lambda Grid size (default 100).
    #' @param lambda_min_ratio Ratio lambda_min/lambda_max; negative (default)
    #'   auto-selects glmnet's rule.
    #' @param seed Fold-permutation RNG seed (default 0).
    #' @param standardize Standardize columns to unit SD (default TRUE).
    #' @param intercept Fit an unpenalized intercept (default TRUE).
    #' @param max_iter Max CD sweeps per lambda (default 100000).
    #' @param tol Convergence tolerance (default 1e-7).
    #' @return Invisibly returns \code{self}.
    fit = function(X, y, alpha = 0.0, n_folds = 10, n_lambda = 100,
                   lambda_min_ratio = -1.0, seed = 0, standardize = TRUE,
                   intercept = TRUE, max_iter = 100000, tol = 1e-7) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(y), nrow(X) == length(y))
      enet_cv_fit(private$cpp_ptr, X, as.numeric(y), as.numeric(alpha),
                  as.integer(n_folds), as.integer(n_lambda),
                  as.numeric(lambda_min_ratio), as.integer(seed),
                  as.logical(standardize), as.logical(intercept),
                  as.integer(max_iter), as.numeric(tol))
      invisible(self)
    },

    #' @description Lambda that minimizes the mean CV error.
    #' @return Scalar numeric.
    cv_min = function() {
      enet_cv_min(private$cpp_ptr)
    },

    #' @description Largest lambda within one SE of the minimum (1SE rule).
    #' @return Scalar numeric.
    cv_1se = function() {
      enet_cv_1se(private$cpp_ptr)
    },

    #' @description Index (1-based) of \code{lambda.min}.
    #' @return Integer.
    index_min = function() {
      enet_cv_index_min(private$cpp_ptr) + 1L
    },

    #' @description Index (1-based) of \code{lambda.1se}.
    #' @return Integer.
    index_1se = function() {
      enet_cv_index_1se(private$cpp_ptr) + 1L
    },

    #' @description Lambda grid used (descending).
    #' @return Numeric vector.
    get_lambdas = function() {
      enet_cv_get_lambdas(private$cpp_ptr)
    },

    #' @description Mean CV MSE per lambda.
    #' @return Numeric vector.
    get_cv_errors = function() {
      enet_cv_get_cv_errors(private$cpp_ptr)
    },

    #' @description Standard error of the CV MSE per lambda.
    #' @return Numeric vector.
    get_cv_std = function() {
      enet_cv_get_cv_std(private$cpp_ptr)
    }
  )
)


# ==============================================================================
# SVDSolver
# ==============================================================================

#' @title SVD Solver
#'
#' @description Computes the top-M truncated SVD of a matrix. Dispatches
#'   internally between a direct (tall matrices), Gram (wide matrices), and
#'   randomized path based on the shape of \code{X}.
#'
#' @details Wraps the C++ \code{SVDSolver::compute(X, M)} static method.
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(50 * 10), 50, 10)
#' solver <- SVDSolver$new()
#' res <- solver$compute(X, M = 3)
#' dim(res$U)  # 50 x 3
#'
#' @export
SVDSolver <- R6::R6Class("SVDSolver",
  public = list(

    #' @description Compute top-M singular value decomposition of \code{X}.
    #'
    #' @param X A numeric matrix (n x p).
    #' @param M Integer, number of singular components to retain.
    #'
    #' @return Named list with:
    #'   \itemize{
    #'     \item \code{U} Left singular vectors (n x M).
    #'     \item \code{S} Singular values (M).
    #'     \item \code{V} Right singular vectors (p x M).
    #'   }
    compute = function(X, M) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(X), is.numeric(M), M >= 1L, M <= min(nrow(X), ncol(X)))
      svd_compute(X, as.integer(M))
    }
  )
)


# ==============================================================================
# PCA
# ==============================================================================

#' @title Principal Component Analysis
#'
#' @description Fits PCA to a data matrix, retaining the top-M principal
#'   components. Centering/normalization is applied to \code{X} in-place at
#'   construction time and undone by \code{restore()}.
#'
#' @details Wraps the C++ \code{PCA} class. The instance is bound to \code{X}
#'   at creation (mirroring the C++ constructor); \code{fit()} is single-use.
#'   The fit result is cached internally; use the \code{get_*()} accessors or
#'   pass \code{X_new} to \code{transform()} for out-of-sample scoring.
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(50 * 8), 50, 8)
#' pca <- PCA$new(X, M = 3, center = TRUE)
#' pca$fit()
#' pca$get_explained_variance()
#' X_new <- matrix(rnorm(10 * 8), 10, 8)
#' pca$transform(X_new)
#' pca$restore()
#'
#' @export
PCA <- R6::R6Class("PCA",
  private = list(
    cpp_ptr = NULL,
    X_      = NULL,
    result_ = NULL
  ),
  public = list(

    #' @description Initialize PCA bound to \code{X}. Centering/normalization
    #'   is applied to \code{X} in-place immediately.
    #'
    #' @param X A numeric matrix (n x p). Modified in-place when
    #'   centering/normalizing; call \code{restore()} to undo.
    #' @param M Integer, number of principal components.
    #' @param center Logical, whether to center \code{X} in-place
    #'   (default: \code{TRUE}).
    #' @param normalize Logical, whether to scale the columns of \code{X}
    #'   in-place (default: \code{TRUE}).
    initialize = function(X, M, center = TRUE, normalize = TRUE) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(X), is.numeric(M), M >= 1L, M <= min(nrow(X), ncol(X)))
      # Double storage so pca_create's zero-copy view aliases this buffer; the
      # same buffer is centered in place here and un-centered by restore().
      private$X_ <- .as_double_matrix(X)
      private$cpp_ptr <- pca_create(private$X_, as.integer(M),
                                    as.logical(center), as.logical(normalize))
    },

    #' @description Fit the PCA bound at construction time (single-use).
    #'
    #' @return Invisible \code{self} for method chaining.
    fit = function() {
      private$result_ <- pca_fit(private$cpp_ptr)
      invisible(self)
    },

    #' @description Project new data onto the fitted principal subspace.
    #'
    #' @param X_new A numeric matrix (n_new x p). The stored training
    #'   means/norms are applied before projection.
    #'
    #' @return Score matrix (n_new x M).
    transform = function(X_new) {
      if (!is.matrix(X_new)) X_new <- as.matrix(X_new)
      stopifnot(!is.null(private$result_), is.numeric(X_new))
      pca_transform(private$cpp_ptr, X_new)
    },

    #' @description Restore \code{X} to its original values (un-center/un-scale).
    #'
    #' @return Invisible \code{self}.
    restore = function() {
      pca_restore(private$cpp_ptr, private$X_)
      invisible(self)
    },

    #' @description Get the PC score matrix from the \code{fit()} call.
    #'
    #' @return Score matrix (n x M), or \code{NULL} if \code{fit()} not yet called.
    get_scores = function() {
      private$result_$Z
    },

    #' @description Get loading matrix from the \code{fit()} call.
    #'
    #' @return Loading matrix (p x M), or \code{NULL} if \code{fit()} not yet called.
    get_loadings = function() {
      private$result_$V
    },

    #' @description Get explained variance per component from the \code{fit()} call.
    #'
    #' @return Numeric vector of length M, or \code{NULL} if \code{fit()} not yet called.
    get_explained_variance = function() {
      private$result_$explained_variance
    },

    #' @description Get the column means used for centering.
    #'
    #' @return Numeric vector of length p. Zero-vector when \code{center = FALSE}.
    get_means = function() {
      pca_get_means(private$cpp_ptr)
    },

    #' @description Get the column scaling factors used for normalization.
    #'
    #' @return Numeric vector of length p. Ones when \code{normalize = FALSE}.
    get_norms = function() {
      pca_get_norms(private$cpp_ptr)
    }
  )
)


# ==============================================================================
# RidgeSolver
# ==============================================================================

#' @title Ridge Regression Solver
#'
#' @description Solves the ridge regression problem
#'   \eqn{\min_\beta \|y - X\beta\|^2 + \lambda \|\beta\|^2}
#'   for a single regularization value. Dispatches internally between primal
#'   (\eqn{n \ge p}) and dual (\eqn{n < p}) Cholesky paths.
#'
#' @details Wraps the C++ \code{RidgeSolver::solve(X, y, lambda)} static method.
#'
#' @examples
#' set.seed(42)
#' n <- 40; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(1, -2, 0, 0.5, 0)
#' y <- X %*% beta_true + rnorm(n, sd = 0.1)
#' solver <- RidgeSolver$new()
#' beta_hat <- solver$solve(X, y, lambda = 0.01)
#'
#' @export
RidgeSolver <- R6::R6Class("RidgeSolver",
  public = list(

    #' @description Solve ridge regression for a single \code{lambda}.
    #'
    #' @param X A numeric matrix (n x p).
    #' @param y Numeric response vector of length n.
    #' @param lambda Non-negative regularization parameter.
    #'
    #' @return Coefficient vector of length p.
    solve = function(X, y, lambda) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(X), is.numeric(y), nrow(X) == length(y),
                is.numeric(lambda), lambda >= 0)
      ridge_solve(X, as.numeric(y), as.numeric(lambda))
    }
  )
)
