#' @title Z-Score Scaler
#'
#' @description Standardizes features by removing the mean and scaling to unit variance.
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(50 * 5), 50, 5)
#' scaler <- ZScoreScaler$new()
#' scaler$fit(X)
#' X_std <- scaler$transform_inplace(X)
#' scaler$get_means()
#'
#' @export
ZScoreScaler <- R6::R6Class("ZScoreScaler",
  private = list(
    cpp_ptr = NULL
  ),
  public = list(

    #' @description Initialize the ZScoreScaler
    #'
    #' @param with_mean Logical, whether to center the data before scaling (default: TRUE)
    #' @param with_std Logical, whether to scale the data to unit variance (default: TRUE)
    initialize = function(with_mean = TRUE, with_std = TRUE) {
      private$cpp_ptr <- zscore_scaler_create(with_mean, with_std)
    },

    #' @description Compute the mean and std to be used for later scaling.
    #'
    #' @param X A numeric matrix (n x p).
    #' @param threshold Double, lower bound for standard deviation (default: 1e-12).
    #'
    #' @return Value.
    fit = function(X, threshold = 1e-12) {
      if (!is.matrix(X)) {
        X <- as.matrix(X)
      }
      zscore_scaler_fit(private$cpp_ptr, X, threshold)
      invisible(self)
    },

    #' @description Perform standardization by centering and scaling in-place.
    #'
    #' @param X A numeric matrix (n x p).
    #'
    #' @return The modified matrix `X`.
    transform_inplace = function(X) {
      if (!is.matrix(X)) {
        X <- as.matrix(X)
      }
      zscore_scaler_transform_inplace(private$cpp_ptr, X)
      return(X)
    },

    #' @description Inverse transform to original scale.
    #'
    #' @param X A numeric matrix (n x p).
    #'
    #' @return The modified matrix `X`.
    inverse_transform_inplace = function(X) {
      if (!is.matrix(X)) {
        X <- as.matrix(X)
      }
      zscore_scaler_inverse_transform_inplace(private$cpp_ptr, X)
      return(X)
    },

    #' @description Check if scaler is fitted.
    #'
    #' @return Value.
    is_fitted = function() {
      zscore_scaler_is_fitted(private$cpp_ptr)
    },

    #' @description Get dropped indices.
    #'
    #' @return Value.
    get_dropped_indices = function() {
      # Convert 0-based Cpp column indices to 1-based R convention
      zscore_scaler_get_dropped_indices(private$cpp_ptr) + 1L
    },

    #' @description Check if scaler centers data.
    #'
    #' @return Value.
    get_with_mean = function() {
      zscore_scaler_get_with_mean(private$cpp_ptr)
    },

    #' @description Check if scaler scales data to unit variance.
    #'
    #' @return Value.
    get_with_std = function() {
      zscore_scaler_get_with_std(private$cpp_ptr)
    },

    #' @description Get the learned means.
    #'
    #' @return Value.
    get_means = function() {
      zscore_scaler_get_means(private$cpp_ptr)
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
#' @description Scales input features to unit norm based on the chosen L1 or L2 formulation.
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
    #' @param with_mean Logical, whether to center the data before scaling (default: TRUE)
    initialize = function(norm_type = 2, with_mean = TRUE) {
      if (!(norm_type %in% c(1, 2))) {
        stop("norm_type must be 1 (L1) or 2 (L2)")
      }
      private$cpp_ptr <- lpnorm_scaler_create(norm_type, with_mean)
    },

    #' @description Compute the norm parameters.
    #'
    #' @param X A numeric matrix (n x p).
    #' @param threshold Double, lower bound for norm (default: 1e-12).
    #'
    #' @return Value.
    fit = function(X, threshold = 1e-12) {
      if (!is.matrix(X)) {
        X <- as.matrix(X)
      }
      lpnorm_scaler_fit(private$cpp_ptr, X, threshold)
      invisible(self)
    },

    #' @description Perform standardization in-place.
    #'
    #' @param X A numeric matrix (n x p).
    #'
    #' @return The modified matrix `X`.
    transform_inplace = function(X) {
      if (!is.matrix(X)) {
        X <- as.matrix(X)
      }
      lpnorm_scaler_transform_inplace(private$cpp_ptr, X)
      return(X)
    },

    #' @description Inverse transform to original scale.
    #'
    #' @param X A numeric matrix (n x p).
    #'
    #' @return The modified matrix `X`.
    inverse_transform_inplace = function(X) {
      if (!is.matrix(X)) {
        X <- as.matrix(X)
      }
      lpnorm_scaler_inverse_transform_inplace(private$cpp_ptr, X)
      return(X)
    },

    #' @description Check if scaler is fitted.
    #'
    #' @return Value.
    is_fitted = function() {
      lpnorm_scaler_is_fitted(private$cpp_ptr)
    },

    #' @description Get dropped indices.
    #'
    #' @return Value.
    get_dropped_indices = function() {
      # Convert 0-based Cpp column indices to 1-based R convention
      lpnorm_scaler_get_dropped_indices(private$cpp_ptr) + 1L
    },

    #' @description Get the learned means.
    #'
    #' @return Value.
    get_means = function() {
      lpnorm_scaler_get_means(private$cpp_ptr)
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
    get_with_mean = function() {
      lpnorm_scaler_get_with_mean(private$cpp_ptr)
    },

    #' @description Check if scaler scales data to unit norm.
    #'
    #' @return Value.
    get_with_norm = function() {
      lpnorm_scaler_get_with_norm(private$cpp_ptr)
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



#' @title Ridge Regression with GCV
#'
#' @description Efficiently solves Ridge Regression while computing Generalized
#'              Cross-Validation (GCV).
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(50 * 5), 50, 5)
#' y <- X[, 1] * 2 + rnorm(50)
#' gcv <- RidgeGCV$new()
#' gcv$fit(X, y)
#' lam_opt <- gcv$gcv_optimal()
#' gcv$solve(lam_opt)
#'
#' @export
RidgeGCV <- R6::R6Class("RidgeGCV",
  private = list(
    cpp_ptr = NULL
  ),
  public = list(
    #' @description Initialize the RidgeGCV object.
    initialize = function() {
      private$cpp_ptr <- ridge_gcv_create()
    },

    #' @description Fit the SVD decomposition for the given data.
    #'
    #' @param X Predictor matrix.
    #' @param y Response vector.
    #'
    #' @return Value.
    fit = function(X, y) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(y), nrow(X) == length(y))
      ridge_gcv_fit(private$cpp_ptr, X, as.numeric(y))
      invisible(self)
    },

    #' @description Solve for a specific lambda.
    #'
    #' @param lambda_val Numeric value for lambda.
    #'
    #' @return Value.
    solve = function(lambda_val) {
      ridge_gcv_solve(private$cpp_ptr, as.numeric(lambda_val))
    },

    #' @description Compute GCV score for a specific lambda.
    #'
    #' @param lambda_val Numeric value for lambda.
    #'
    #' @return Value.
    gcv = function(lambda_val) {
      ridge_gcv_gcv_score(private$cpp_ptr, as.numeric(lambda_val))
    },

    #' @description Predict using a specific lambda.
    #'
    #' @param X_new Numeric matrix for predictions.
    #' @param lambda_val Numeric value for lambda.
    #'
    #' @return Value.
    predict = function(X_new, lambda_val) {
      if (!is.matrix(X_new)) X_new <- as.matrix(X_new)
      ridge_gcv_predict(private$cpp_ptr, X_new, as.numeric(lambda_val))
    },

    #' @description Predict using a path of lambdas.
    #'
    #' @param X_new Numeric matrix for predictions.
    #' @param lambdas Numeric vector of lambda values.
    #'
    #' @return Value.
    predict_path = function(X_new, lambdas) {
      if (!is.matrix(X_new)) X_new <- as.matrix(X_new)
      ridge_gcv_predict_path(private$cpp_ptr, X_new, as.numeric(lambdas))
    },

    #' @description Solve across a path of lambdas.
    #'
    #' @param lambdas Numeric vector of lambda values.
    #'
    #' @return Value.
    solve_path = function(lambdas) {
      result <- ridge_gcv_solve_path(private$cpp_ptr, as.numeric(lambdas))
      # Convert 0-based Cpp best_index to 1-based R convention
      result$best_index <- result$best_index + 1L
      result
    },

    #' @description Find the optimal lambda.
    #'
    #' @param num_grid Integer, number of grid points.
    #' @param tol Numeric, tolerance for convergence.
    #'
    #' @return Value.
    gcv_optimal = function(num_grid = 200, tol = 1e-8) {
      ridge_gcv_optimal(private$cpp_ptr, as.integer(num_grid), as.numeric(tol))
    },

    #' @description Get the default lambda grid.
    #'
    #' @param num_grid Integer, number of grid points.
    #' @param ratio Numeric, ratio for lambda grid.
    #'
    #' @return Value.
    default_lambda_grid = function(num_grid = 100, ratio = 1000) {
      ridge_gcv_default_lambda_grid(private$cpp_ptr, as.integer(num_grid), as.numeric(ratio))
    },

    #' @description Get number of samples.
    #'
    #' @return Value.
    n_samples = function() {
      ridge_gcv_n_samples(private$cpp_ptr)
    },

    #' @description Get number of features.
    #'
    #' @return Value.
    n_features = function() {
      ridge_gcv_n_features(private$cpp_ptr)
    },

    #' @description Get rank of X.
    #'
    #' @return Value.
    rank = function() {
      ridge_gcv_rank(private$cpp_ptr)
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
#'   components. Optionally centers \code{X} in-place during \code{fit()} and
#'   restores it on \code{restore()} or when the object is garbage-collected.
#'
#' @details Wraps the C++ \code{PCA} class with RAII centering semantics.
#'   The fit result is cached internally; use the \code{get_*()} accessors or
#'   pass \code{X_new} to \code{transform()} for out-of-sample scoring.
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(50 * 8), 50, 8)
#' pca <- PCA$new(center = TRUE)
#' pca$fit(X, M = 3)
#' pca$get_explained_variance()
#' X_new <- matrix(rnorm(10 * 8), 10, 8)
#' pca$transform(X_new)
#' pca$restore()
#'
#' @export
PCA <- R6::R6Class("PCA",
  private = list(
    cpp_ptr = NULL,
    result_  = NULL
  ),
  public = list(

    #' @description Initialize PCA.
    #'
    #' @param center Logical, whether to center \code{X} in-place during
    #'   \code{fit()} (default: \code{TRUE}).
    initialize = function(center = TRUE) {
      private$cpp_ptr <- pca_create(as.logical(center))
    },

    #' @description Fit PCA to \code{X}, retaining the top \code{M} components.
    #'
    #' @details When \code{center = TRUE}, \code{X} is modified in-place (column
    #'   means subtracted). Call \code{restore()} to undo this.
    #'
    #' @param X A numeric matrix (n x p). Modified in-place when centering.
    #' @param M Integer, number of principal components.
    #'
    #' @return Invisible \code{self} for method chaining.
    fit = function(X, M) {
      if (!is.matrix(X)) X <- as.matrix(X)
      stopifnot(is.numeric(X), is.numeric(M), M >= 1L, M <= min(nrow(X), ncol(X)))
      private$result_ <- pca_fit(private$cpp_ptr, X, as.integer(M))
      invisible(self)
    },

    #' @description Project new data onto the fitted principal subspace.
    #'
    #' @param X_new A numeric matrix (n_new x p). Column means are subtracted
    #'   using the stored training mean before projection.
    #'
    #' @return Score matrix (n_new x M).
    transform = function(X_new) {
      if (!is.matrix(X_new)) X_new <- as.matrix(X_new)
      stopifnot(!is.null(private$result_), is.numeric(X_new))
      pca_transform(private$cpp_ptr, X_new)
    },

    #' @description Restore \code{X} to its original values (un-center).
    #'
    #' @return Invisible \code{self}.
    restore = function() {
      pca_restore(private$cpp_ptr)
      invisible(self)
    },

    #' @description Get the PC score matrix from the last \code{fit()} call.
    #'
    #' @return Score matrix (n x M), or \code{NULL} if \code{fit()} not yet called.
    get_scores = function() {
      private$result_$Z
    },

    #' @description Get loading matrix from the last \code{fit()} call.
    #'
    #' @return Loading matrix (p x M), or \code{NULL} if \code{fit()} not yet called.
    get_loadings = function() {
      private$result_$V
    },

    #' @description Get explained variance per component from the last \code{fit()} call.
    #'
    #' @return Numeric vector of length M, or \code{NULL} if \code{fit()} not yet called.
    get_explained_variance = function() {
      private$result_$explained_variance
    },

    #' @description Get the column means used for centering.
    #'
    #' @return Numeric vector of length p. Zero-vector when \code{center = FALSE}.
    get_mean = function() {
      pca_get_mean(private$cpp_ptr)
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
