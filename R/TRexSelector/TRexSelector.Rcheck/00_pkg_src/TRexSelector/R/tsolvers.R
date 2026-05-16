#' @useDynLib TRexSelector
#' @importFrom Rcpp evalCpp
#' @import R6
NULL

#' Base T-Solver Class
#'
#' @description Wraps the underlying C++ TSolver instances over R6.
#' @field X Numeric matrix of predictors.
#' @field D Numeric matrix of dummies/features.
#' @field y Numeric vector of response variable.
#' @field ptr External pointer to C++ object.
#' @export
TSolver <- R6::R6Class(
  "TSolver",
  public = list(
    X = NULL,
    D = NULL,
    y = NULL,
    ptr = NULL,

    #' @description Method description.
    #' @param solver_type Character, type of solver to use.
    #' @param X Numeric matrix of predictors.
    #' @param D Numeric matrix of dummies/features.
    #' @param y Numeric vector of response variable.
    initialize = function(solver_type = "lars", X, D, y) {
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

      if (solver_type == "lars") {
        if (is_mmap) {
          self$ptr <- tsolver_lars_mmap_create(self$X, self$D, self$y)
        } else {
          self$ptr <- tsolver_lars_create(self$X, self$D, self$y)
        }
      } else if (solver_type == "omp") {
        if (is_mmap) {
          self$ptr <- tsolver_omp_mmap_create(self$X, self$D, self$y)
        } else {
          self$ptr <- tsolver_omp_create(self$X, self$D, self$y)
        }
      } else {
        stop("Unknown solver_type: ", solver_type)
      }
    },

    #' @description Method description.
    #' @param T_stop Integer, stopping criterion.
    #' @param early_stop Logical, whether to stop early.
    #' @return Value.
    execute_step = function(T_stop = -1, early_stop = TRUE) {
      tsolver_execute_step(self$ptr, T_stop, early_stop)
      invisible(self)
    },

    #' @description Method description.
    #' @param step Integer, step number.
    #' @return Value.
    get_beta = function(step = -1) {
      tsolver_get_beta(self$ptr, step)
    },

    #' @description Method description.
    #' @return Value.
    get_cp = function() {
      tsolver_get_cp(self$ptr)
    },

    #' @description Method description.
    #' @return Value.
    get_rss = function() {
      tsolver_get_rss(self$ptr)
    },

    #' @description Method description.
    #' @return Value.
    get_r2 = function() {
      tsolver_get_r2(self$ptr)
    },

    #' @description Method description.
    #' @return Value.
    get_dof = function() {
      tsolver_get_dof(self$ptr)
    },

    #' @description Method description.
    #' @param r_index Integer, starting from 1 in R.
    #' @return Value.
    get_actives = function(r_index = TRUE) {
      # Add 1 if requesting 1-based R indexes (default for R APIs)
      res <- tsolver_get_actives(self$ptr)
      if (r_index) {
        res <- res + 1
      }
      res
    },

    #' @description Method description.
    #' @param r_index Integer, starting from 1 in R.
    #' @return Value.
    get_inactives = function(r_index = TRUE) {
      res <- tsolver_get_inactives(self$ptr)
      if (r_index) {
        res <- res + 1
      }
      res
    },

    #' @description Method description.
    #' @param step Integer, step number.
    #' @return Value.
    get_intercept = function(step = -1) {
      tsolver_get_intercept(self$ptr, step)
    }
  )
)
