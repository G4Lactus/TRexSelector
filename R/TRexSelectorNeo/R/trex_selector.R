# =========================================================================
# trex_selector.R - TRexSelector class definition
# =========================================================================

#' @name TRexSelector
#'
#' @title T-Rex Selector
#'
#' @description FDR-controlled variable selection algorithm.
#'
#' @importFrom R6 R6Class
#'
#' @examples
#' \donttest{
#' data(Gauss_data)
#' X <- Gauss_data$X
#' y <- Gauss_data$y
#' sel <- TRexSelector$new(X, y, tFDR = 0.1, verbose = FALSE)
#' sel$select()
#' sel$selected_indices
#' }
#' @export
TRexSelector <- R6::R6Class("TRexSelector",
  public = list(
    #' @description Create a new TRexSelector object.
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param tFDR Target FDR level (default: 0.1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param control A control list from \code{\link{trex_control}()} (default:
    #'   \code{trex_control()}).
    initialize = function(X, y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          control = trex_control()) {

      if (inherits(X, "MemoryMappedMatrix")) {
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_selector_mmap_create(
          get_ptr(X), y, tFDR, control, seed, verbose
        )
      } else {
        X <- .as_double_matrix(X)
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_selector_create(
          X, y, tFDR, control, seed, verbose
        )
      }
    },

    #' @description Run the TRexSelector algorithm.
    #' @return invisible(self)
    select = function() {
      trex_selector_select(private$ptr)
      invisible(self)
    }
  ),
  active = list(
    #' @field .ptr Get internal pointer.
    .ptr = function(value) {
      if (!missing(value)) stop(".ptr is read-only")
      private$ptr
    },

    #' @field .ptr_X internal proxy for data size
    .ptr_X = function(value) {
      if (!missing(value)) stop(".ptr_X read-only")
      private$refs$X
    },

    #' @field selected_var Logical vector indicating which variables were selected.
    selected_var = function(value) {
      if (!missing(value)) stop("selected_var is read-only.")
      trex_selector_get_selected_var(private$ptr)
    },

    #' @field selected_indices Integer vector of selected 1-based indices.
    selected_indices = function(value) {
      if (!missing(value)) stop("selected_indices is read-only.")
      trex_selector_get_selected_indices(private$ptr)
    },

    #' @field v_thresh The chosen voting threshold.
    v_thresh = function(value) {
      if (!missing(value)) stop("v_thresh is read-only.")
      trex_selector_get_voting_threshold(private$ptr)
    },

    #' @field T_stop The length of the path / maximum number of included variables.
    T_stop = function(value) {
      if (!missing(value)) stop("T_stop is read-only.")
      trex_selector_get_t_stop(private$ptr)
    },

    #' @field L The number of dummy replications used.
    L = function(value) {
      if (!missing(value)) stop("L is read-only.")
      trex_selector_get_dummy_multiplier_l(private$ptr)
    },

    #' @field voting_grid The grid of voters over trials and variables.
    voting_grid = function(value) {
      if (!missing(value)) stop("voting_grid is read-only.")
      trex_selector_get_voting_grid(private$ptr)
    },

    #' @field fdp_hat_mat estimated false discovery proportions
    fdp_hat_mat = function(value) {
      if (!missing(value)) stop("fdp_hat_mat is read-only")
      trex_selector_get_fdp_hat_mat(private$ptr)
    },

    #' @field phi_mat probability mass function evaluations
    phi_mat = function(value) {
      if (!missing(value)) stop("phi_mat is read-only")
      trex_selector_get_phi_mat(private$ptr)
    },

    #' @field r_mat empirical CDF computations
    r_mat = function(value) {
      if (!missing(value)) stop("r_mat is read-only")
      trex_selector_get_r_mat(private$ptr)
    },

    #' @field phi_prime smoothed probability masses
    phi_prime = function(value) {
      if (!missing(value)) stop("phi_prime is read-only")
      trex_selector_get_phi_prime(private$ptr)
    },

    #' @field x_means computed means of the predictor variables
    x_means = function(value) {
      if (!missing(value)) stop("x_means is read-only")
      trex_selector_get_x_means(private$ptr)
    },

    #' @field x_l2_norms computed L2 norms of the scaled predictors
    x_l2_norms = function(value) {
      if (!missing(value)) stop("x_l2_norms is read-only")
      trex_selector_get_x_l2_norms(private$ptr)
    },

    #' @field y_mean computed mean of the response sequence
    y_mean = function(value) {
      if (!missing(value)) stop("y_mean is read-only")
      trex_selector_get_y_mean(private$ptr)
    }
  ),
  private = list(
    ptr = NULL,
    refs = NULL
  )
)
