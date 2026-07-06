# =============================================================================
# tsolvers.R - T-Solver R6 class definitions
# =============================================================================


#' @importFrom Rcpp evalCpp
#' @import R6
NULL

# ==============================================================================
# TSolver_Base
# ==============================================================================

#' Base T-Solver Class
#'
#' @description
#' Abstract base class for all T-Solver R6 wrappers. Provides the common solver
#' API backed by a Cpp \code{TSolver_Base} instance. Users should instantiate one
#' of the concrete subclasses (\code{TLARS_Solver}, \code{TLASSO_Solver}, etc.)
#' rather than this class directly.
#'
#' @field X Numeric matrix of predictors, or an \code{mmap_matrix}.
#' @field D Numeric matrix of dummies.
#' @field y Numeric response vector.
#' @field ptr External pointer to the underlying Cpp \code{TSolver_Base} object.
#' @examples
#' # TSolver_Base is abstract; use a concrete subclass instead, e.g.:
#' # solver <- TLARS_Solver$new(X, D, y)
#' @export
TSolver_Base <- R6::R6Class(
  "TSolver_Base",
  public = list(
    X   = NULL,
    D   = NULL,
    y   = NULL,
    ptr = NULL,

    #' @description
    #' Validate and store data fields. Called by every subclass \code{initialize}.
    #' Not intended for direct instantiation.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    initialize = function(X, D, y) {
      is_mmap <- inherits(X, "mmap_matrix")
      if (!is.matrix(X) && !is_mmap) {
        stop("Input X must be a numeric matrix or mmap_matrix.")
      }
      if (!is.matrix(D) || !is.numeric(y)) {
        stop("Inputs D must be a matrix, and y must be numeric.")
      }
      self$X <- X
      self$D <- D
      self$y <- as.vector(y)
    },

    #' @description Execute solver step.
    #' @param T_stop Integer stopping criterion; \code{-1} runs until the
    #'   early-stop condition is met.
    #' @param early_stop Logical; stop early when enough dummies are active.
    #' @return Invisibly returns \code{self} for method chaining.
    execute_step = function(T_stop = -1, early_stop = TRUE) {
      tsolver_execute_step(self$ptr, T_stop, early_stop)
      invisible(self)
    },

    #' @description Get the number of steps executed so far.
    #' @return Integer number of completed steps.
    get_num_steps = function() {
      tsolver_get_num_steps(self$ptr)
    },

    #' @description Get the action path (variable additions and removals per step).
    #' @param r_index Logical; if \code{TRUE} (default) indices inside each
    #'   action vector are 1-based (R convention); if \code{FALSE} they are
    #'   0-based (Cpp convention).
    #' @return A list of integer vectors, one per step. Positive indices denote
    #'   variable additions; negative indices denote removals.
    get_actions = function(r_index = TRUE) {
      acts <- tsolver_get_actions(self$ptr)
      if (r_index) {
        lapply(acts, function(a) ifelse(a >= 0L, a + 1L, a - 1L))
      } else {
        acts
      }
    },

    #' @description Get coefficient vector at a given step.
    #' @param step Integer step index; \code{-1} returns the current/final step.
    #' @return Numeric vector of coefficients.
    get_beta = function(step = -1) {
      b <- tsolver_get_beta(self$ptr, step)
      b[seq_len(ncol(self$X))]
    },

    #' @description Get Mallows Cp scores along the solution path.
    #' @return Numeric vector of Cp scores, one per step.
    get_cp = function() {
      tsolver_get_cp(self$ptr)[-1]
    },

    #' @description Get residual sum of squares along the solution path.
    #' @return Numeric vector of RSS values, one per step.
    get_rss = function() {
      tsolver_get_rss(self$ptr)[-1]
    },

    #' @description Get R-squared values along the solution path.
    #' @return Numeric vector of R-squared values, one per step.
    get_r2 = function() {
      tsolver_get_r2(self$ptr)[-1]
    },

    #' @description Get degrees of freedom along the solution path.
    #' @return Numeric vector of degrees of freedom, one per step.
    get_dof = function() {
      tsolver_get_dof(self$ptr)[-1]
    },

    #' @description Get currently active variable indices.
    #' @param r_index Logical; if \code{TRUE} (default) indices are 1-based
    #'   (R convention); if \code{FALSE} they are 0-based (Cpp convention).
    #' @return Integer vector of active variable indices.
    get_actives = function(r_index = TRUE) {
      p <- ncol(self$X)
      res <- tsolver_get_actives(self$ptr)
      res <- res[res < p]
      if (r_index) res <- res + 1L
      res
    },

    #' @description Get currently inactive variable indices.
    #' @param r_index Logical; if \code{TRUE} (default) indices are 1-based
    #'   (R convention); if \code{FALSE} they are 0-based (Cpp convention).
    #' @return Integer vector of inactive variable indices.
    get_inactives = function(r_index = TRUE) {
      p <- ncol(self$X)
      res <- tsolver_get_inactives(self$ptr)
      res <- res[res < p]
      if (r_index) res <- res + 1L
      res
    },

    #' @description Get the fitted intercept at a given step.
    #' @param step Integer step index; \code{-1} returns the current/final step.
    #' @return Scalar numeric intercept.
    get_intercept = function(step = -1) {
      tsolver_get_intercept(self$ptr, step)
    },

    #' @description Serialize the solver state to a binary file.
    #' @param filename Character; path to the output file.
    #' @return Invisibly returns \code{self}.
    save = function(filename) {
      tsolver_save(self$ptr, filename)
      invisible(self)
    },

    #' @description Restore the solver state from a binary file created by
    #'   \code{$save()}.
    #' @param filename Character; path to the input file.
    #' @return Invisibly returns \code{self}.
    load = function(filename) {
      tsolver_load(self$ptr, filename)
      invisible(self)
    }
  )
)


# ==============================================================================
# LARS-family subclasses
# ==============================================================================

#' T-LARS Solver
#'
#' @description
#' Terminating Least Angle Regression (T-LARS) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TLARS_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#' solver$get_actives()
#'
#' @export
TLARS_Solver <- R6::R6Class(
  "TLARS_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TLARS_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_lars_mmap_create(self$X, self$D, self$y,
                                             normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_lars_create(self$X, self$D, self$y,
                                        normalize, intercept, verbose)
      }
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    }
  )
)


#' T-LASSO Solver
#'
#' @description
#' Terminating LASSO solver (T-LARS with variable removal).
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TLASSO_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TLASSO_Solver <- R6::R6Class(
  "TLASSO_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TLASSO_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tlasso_mmap_create(self$X, self$D, self$y,
                                               normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tlasso_create(self$X, self$D, self$y,
                                          normalize, intercept, verbose)
      }
    },

    #' @description Get the number of coefficient removals.
    #' @return Integer count of removals.
    get_num_removals = function() {
      tsolver_get_num_removals(self$ptr)
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    }
  )
)


#' T-Stepwise Solver
#'
#' @description
#' Terminating Forward Stepwise Regression solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TSTEPWISE_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TSTEPWISE_Solver <- R6::R6Class(
  "TSTEPWISE_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TSTEPWISE_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tstepwise_mmap_create(self$X, self$D, self$y,
                                                  normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tstepwise_create(self$X, self$D, self$y,
                                             normalize, intercept, verbose)
      }
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    }
  )
)


#' T-ENET Solver
#'
#' @description
#' Terminating Elastic Net solver (L1 + L2 regularization).
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TENET_Solver$new(X, D, y, lambda2 = 0.1)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TENET_Solver <- R6::R6Class(
  "TENET_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TENET_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param lambda2 Numeric L2 (ridge) penalty parameter (>= 0).
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          lambda2,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tenet_mmap_create(self$X, self$D, self$y,
                                              lambda2, normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tenet_create(self$X, self$D, self$y,
                                         lambda2, normalize, intercept, verbose)
      }
    },

    #' @description Get the number of coefficient removals.
    #'
    #' @return Integer count of removals.
    get_num_removals = function() {
      tsolver_get_num_removals(self$ptr)
    },

    #' @description Get the cycling ratio (removals / total steps).
    #'
    #' @return Scalar numeric cycling ratio.
    get_cycling_ratio = function() {
      tsolver_get_cycling_ratio(self$ptr)
    },

    #' @description Get the regularization (lambda) path.
    #'
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    }
  )
)


#' T-ENET-Aug Solver
#'
#' @description
#' Terminating Elastic Net solver via row augmentation (TENETAug). The ridge
#' penalty is encoded in an augmented data system solved by an inner T-LASSO
#' (or pure T-LARS) path, matching the R reference implementation of
#' \code{trex_GVS()}. Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TENETAug_Solver$new(X, D, y, lambda2 = 0.1)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TENETAug_Solver <- R6::R6Class(
  "TENETAug_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TENETAug_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param lambda2 Numeric L2 (ridge) penalty parameter (>= 0).
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    #' @param use_lars_inner Logical; run the inner path with a pure-LARS
    #'   solver that never drops variables (default \code{FALSE}).
    initialize = function(X, D, y,
                          lambda2,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE,
                          use_lars_inner = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tenet_aug_mmap_create(self$X, self$D, self$y,
                                                  lambda2, normalize, intercept,
                                                  verbose, use_lars_inner)
      } else {
        self$ptr <- tsolver_tenet_aug_create(self$X, self$D, self$y,
                                             lambda2, normalize, intercept,
                                             verbose, use_lars_inner)
      }
    }
  )
)


#' T-IENET-Aug Solver
#'
#' @description
#' Terminating Informed Elastic Net solver via row augmentation (TIENETAug).
#' The group-ridge penalty (one augmented row per group, scaled by
#' \code{sqrt(lambda2)}) is encoded in an augmented data system solved by an
#' inner T-LASSO (or pure T-LARS) path. This is the solver used internally by
#' \code{TRexGVSSelector} for \code{gvs_type = "IEN"}.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' groups <- c(1, 1, 2, 2)  # 1-based group id per predictor
#' solver <- TIENETAug_Solver$new(X, D, y, lambda2 = 0.1, groups = groups)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TIENETAug_Solver <- R6::R6Class(
  "TIENETAug_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TIENETAug_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param lambda2 Numeric group-ridge penalty parameter (>= 0).
    #' @param groups Integer vector of 1-based group ids, one per predictor
    #'   column of \code{X} (contiguous ids starting at 1).
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    #' @param use_lars_inner Logical; run the inner path with a pure-LARS
    #'   solver that never drops variables (default \code{FALSE}).
    initialize = function(X, D, y,
                          lambda2,
                          groups,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE,
                          use_lars_inner = FALSE) {
      super$initialize(X, D, y)
      groups <- as.integer(groups)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tienet_aug_mmap_create(self$X, self$D, self$y,
                                                   lambda2, groups, normalize,
                                                   intercept, verbose,
                                                   use_lars_inner)
      } else {
        self$ptr <- tsolver_tienet_aug_create(self$X, self$D, self$y,
                                              lambda2, groups, normalize,
                                              intercept, verbose,
                                              use_lars_inner)
      }
    },

    #' @description Get the group-ridge penalty lambda2 used.
    #' @return Scalar numeric lambda2.
    get_lambda2 = function() {
      tsolver_tienet_aug_get_lambda2(self$ptr)
    },

    #' @description Get the group ids (1-based, one per predictor).
    #' @return Integer vector of group ids.
    get_groups = function() {
      tsolver_tienet_aug_get_groups(self$ptr)
    },

    #' @description Get the number of disjoint groups.
    #' @return Integer group count.
    get_num_groups = function() {
      tsolver_tienet_aug_get_num_groups(self$ptr)
    }
  )
)


#' T-Stagewise Solver
#'
#' @description
#' Terminating Stagewise Regression solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TSTAGEWISE_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TSTAGEWISE_Solver <- R6::R6Class(
  "TSTAGEWISE_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TSTAGEWISE_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tstagewise_mmap_create(self$X, self$D, self$y,
                                                   normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tstagewise_create(self$X, self$D, self$y,
                                              normalize, intercept, verbose)
      }
    },

    #' @description Get the number of coefficient removals.
    #' @return Integer count of removals.
    get_num_removals = function() {
      tsolver_get_num_removals(self$ptr)
    },

    #' @description Get the cycling ratio (removals / total steps).
    #' @return Scalar numeric cycling ratio.
    get_cycling_ratio = function() {
      tsolver_get_cycling_ratio(self$ptr)
    },

    #' @description Get the regularization (lambda) path.
    #' @return Numeric vector of lambda values, one per step.
    get_lambda = function() {
      tsolver_get_lambda(self$ptr)
    }
  )
)


# ==============================================================================
# OMP-family subclasses
# ==============================================================================

#' T-OMP Solver
#'
#' @description
#' Terminating Orthogonal Matching Pursuit (T-OMP) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TOMP_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TOMP_Solver <- R6::R6Class(
  "TOMP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TOMP_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_omp_mmap_create(self$X, self$D, self$y,
                                            normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_omp_create(self$X, self$D, self$y,
                                       normalize, intercept, verbose)
      }
    }
  )
)


#' T-GP Solver
#'
#' @description
#' Terminating Gradient Pursuit (T-GP) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TGP_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TGP_Solver <- R6::R6Class(
  "TGP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TGP_Solver}.
    #'
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tgp_mmap_create(self$X, self$D, self$y,
                                            normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tgp_create(self$X, self$D, self$y,
                                       normalize, intercept, verbose)
      }
    }
  )
)


#' T-ACGP Solver
#'
#' @description
#' Terminating Approximate Conjugate Gradient Pursuit (T-ACGP) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TACGP_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TACGP_Solver <- R6::R6Class(
  "TACGP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TACGP_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tacgp_mmap_create(self$X, self$D, self$y,
                                              normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tacgp_create(self$X, self$D, self$y,
                                         normalize, intercept, verbose)
      }
    }
  )
)


#' T-MP Solver
#'
#' @description
#' Terminating Matching Pursuit (T-MP) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TMP_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TMP_Solver <- R6::R6Class(
  "TMP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TMP_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tmp_mmap_create(self$X, self$D, self$y,
                                            normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tmp_create(self$X, self$D, self$y,
                                       normalize, intercept, verbose)
      }
    }
  )
)


#' T-OLS Solver
#'
#' @description
#' Terminating Orthogonal Least Squares (T-OLS) solver.
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TOOLS_Solver$new(X, D, y)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TOOLS_Solver <- R6::R6Class(
  "TOOLS_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TOOLS_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tools_mmap_create(self$X, self$D, self$y,
                                              normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tools_create(self$X, self$D, self$y,
                                         normalize, intercept, verbose)
      }
    }
  )
)


#' T-NCGMP Solver
#'
#' @description
#' Terminating Norm-Corrected Gradient Matching Pursuit (T-NCGMP) solver.
#' Two variants are available: \code{"line_search"} (standard, 1-D projection)
#' and \code{"fully_corrective"} (OMP-style least squares over the active set).
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TNCGMP_Solver$new(X, D, y, variant = "line_search")
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TNCGMP_Solver <- R6::R6Class(
  "TNCGMP_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TNCGMP_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param variant Character; \code{"line_search"} (default) or
    #'   \code{"fully_corrective"}.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          variant   = "line_search",
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      v <- switch(variant,
        "line_search"      = 0L,
        "fully_corrective" = 1L,
        stop("TNCGMP_Solver: unknown variant '", variant,
             "'. Use \"line_search\" or \"fully_corrective\".")
      )
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tncgmp_mmap_create(self$X, self$D, self$y,
                                               v, normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tncgmp_create(self$X, self$D, self$y,
                                          v, normalize, intercept, verbose)
      }
    }
  )
)


# ==============================================================================
# AFS-family subclasses
# ==============================================================================

#' T-AFS Solver
#'
#' @description
#' Terminating Adaptive Forward Stepwise (T-AFS) solver.
#' The shrinkage parameter \code{rho} interpolates between Forward Stepwise
#' (\code{rho = 1}) and LAR/LASSO (\code{rho -> 0}).
#' Inherits all common methods from \code{\link{TSolver_Base}}.
#'
#' @examples
#' set.seed(1)
#' n <- 20; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' D <- matrix(rnorm(n * p), n, p)
#' y <- X[, 1] * 2 + rnorm(n)
#' solver <- TAFS_Solver$new(X, D, y, rho = 0.5)
#' solver$execute_step(T_stop = 3)
#' solver$get_beta()
#'
#' @export
TAFS_Solver <- R6::R6Class(
  "TAFS_Solver",
  inherit = TSolver_Base,
  public = list(
    #' @description Create a new \code{TAFS_Solver}.
    #' @param X Numeric matrix of predictors, or \code{mmap_matrix}.
    #' @param D Numeric matrix of dummies.
    #' @param y Numeric response vector.
    #' @param rho Numeric shrinkage parameter in (0, 1]; default \code{1.0}.
    #' @param normalize Logical; column-normalize X and D (default \code{TRUE}).
    #' @param intercept Logical; fit an intercept (default \code{TRUE}).
    #' @param verbose Logical; print progress (default \code{FALSE}).
    initialize = function(X, D, y,
                          rho       = 1.0,
                          normalize = TRUE,
                          intercept = TRUE,
                          verbose   = FALSE) {
      super$initialize(X, D, y)
      if (inherits(self$X, "mmap_matrix")) {
        self$ptr <- tsolver_tafs_mmap_create(self$X, self$D, self$y,
                                             rho, normalize, intercept, verbose)
      } else {
        self$ptr <- tsolver_tafs_create(self$X, self$D, self$y,
                                        rho, normalize, intercept, verbose)
      }
    }
  )
)
