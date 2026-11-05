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
#' @references
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
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
#' @family T-Rex selectors and controls
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

    #' @description Give \code{X} back before \code{select()}: undo the
    #'   in-place normalization applied at construction and release the
    #'   buffer claim (idempotent; a no-op after \code{select()}, which
    #'   restores \code{X} itself). A restored selector cannot run:
    #'   \code{select()} then errors, construct a new one. Without it, an
    #'   unselected selector gives \code{X} back only when it is
    #'   garbage-collected.
    #' @return Invisibly returns \code{self}.
    restore = function() {
      trex_selector_restore(private$ptr)
      invisible(self)
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

    #' @field T_stop Stopping time: the number of included dummies at which
    #'   the T-loop stopped.
    T_stop = function(value) {
      if (!missing(value)) stop("T_stop is read-only.")
      trex_selector_get_t_stop(private$ptr)
    },

    #' @field L Calibrated dummy multiplier L (the selector uses
    #'   L * p dummies).
    L = function(value) {
      if (!missing(value)) stop("L is read-only.")
      trex_selector_get_dummy_multiplier_l(private$ptr)
    },

    #' @field voting_grid Voting levels 0.5, 0.5 + 1/K, ..., 1 - eps that the
    #'   selection searches.
    voting_grid = function(value) {
      if (!missing(value)) stop("voting_grid is read-only.")
      trex_selector_get_voting_grid(private$ptr)
    },

    #' @field fdp_hat_mat Estimated false discovery proportion per T step
    #'   (rows) and voting level (columns).
    fdp_hat_mat = function(value) {
      if (!missing(value)) stop("fdp_hat_mat is read-only")
      trex_selector_get_fdp_hat_mat(private$ptr)
    },

    #' @field phi_mat Relative occurrences of the original variables over
    #'   the K random experiments after each T step (rows).
    phi_mat = function(value) {
      if (!missing(value)) stop("phi_mat is read-only")
      trex_selector_get_phi_mat(private$ptr)
    },

    #' @field r_mat Number of selected variables per T step (rows) and
    #'   voting level (columns).
    r_mat = function(value) {
      if (!missing(value)) stop("r_mat is read-only")
      trex_selector_get_r_mat(private$ptr)
    },

    #' @field phi_prime Calibrated relative occurrences used by the FDP
    #'   estimate (length p), at most 1 and possibly negative.
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
    },

    #' @field tFDR The target FDR level.
    tFDR = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      trex_selector_get_tfdr(private$ptr)
    },

    #' @field K The number of random experiments.
    K = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      trex_selector_get_k(private$ptr)
    },

    #' @field max_num_dummies The maximum dummy multiplier of the L-loop.
    max_num_dummies = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      trex_selector_get_max_num_dummies(private$ptr)
    },

    #' @field max_T_stop Whether the maximum T_stop rule is active.
    max_T_stop = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      trex_selector_get_max_t_stop(private$ptr)
    },

    #' @field stagnation_check Whether the stagnation check is active.
    stagnation_check = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      trex_selector_get_stagnation_check(private$ptr)
    },

    #' @field num_selected The number of selected variables.
    num_selected = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      trex_selector_get_num_selected(private$ptr)
    },

    #' @field num_dummies The number of dummies used by the last select()
    #'   (the L-loop's calibration).
    num_dummies = function(value) {
      if (!missing(value)) stop("Field is read-only.")
      trex_selector_get_num_dummies(private$ptr)
    }
  ),
  private = list(
    ptr = NULL,
    refs = NULL
  )
)
