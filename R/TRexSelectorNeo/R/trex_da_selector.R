#' @name TRexDASelector
#'
#' @title Dependency-Aware T-Rex Selector (TRexDA)
#'
#' @description FDR-controlled variable selection algorithm for dependency scenarios.
#'
#' @importFrom R6 R6Class
#'
#' @examples
#' \donttest{
#' data(Gauss_data)
#' X <- Gauss_data$X
#' y <- Gauss_data$y
#' sel <- TRexDASelector$new(X, y, tFDR = 0.1, verbose = FALSE,
#'                           da_control = trex_da_control(da_method = "BT"))
#' sel$select()
#' sel$selected_indices
#' }
#'
#' @export
TRexDASelector <- R6::R6Class("TRexDASelector",
  inherit = TRexSelector,
  public = list(
    #' @description Create a new TRexDASelector object.
    #'
    #' @param X Feature matrix or MemoryMappedMatrix.
    #' @param y Response vector.
    #' @param tFDR Target FDR level (default: 0.1).
    #' @param seed Random seed (default: -1).
    #' @param verbose Whether to print progress (default: TRUE).
    #' @param da_control A DA control list from \code{\link{trex_da_control}()} (default:
    #'   \code{trex_da_control()}).
    #' @param control A control list from \code{\link{trex_control}()} (default:
    #'   \code{trex_control()}).
    initialize = function(X, y,
                          tFDR = 0.1,
                          seed = -1,
                          verbose = TRUE,
                          da_control = trex_da_control(),
                          control = trex_control()) {

      # No need to call super$initialize() because the Cpp pointer constructor is different.
      if (inherits(X, "MemoryMappedMatrix")) {
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_da_mmap_create(
          get_ptr(X), y, tFDR, da_control, control, seed, verbose
        )
      } else {
        X <- .as_double_matrix(X)
        y <- .as_double_vector(y)
        private$refs <- list(X = X, y = y)
        private$ptr <- trex_da_create(
          X, y, tFDR, da_control, control, seed, verbose
        )
      }
    },

    #' @description Run the Dependency-Aware TRexSelector algorithm.
    select = function() {
      trex_da_select(private$ptr)
      invisible(self)
    }
  ),
  active = list(
    #' @field rho_thresh Optimal correlation threshold selected by the DA process.
    rho_thresh = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_da_get_rho_thresh(private$ptr)
    },
    #' @field rho_grid The grid of correlation thresholds evaluated.
    rho_grid = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_da_get_rho_grid(private$ptr)
    },

    #' @field da_method_used The integer code (or name mapped later) of the DA method used.
    da_method_used = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      trex_da_get_method_string(private$ptr)
    },

    #' @field FDP_hat_array_BT A 3D array of FDP estimates for BT method:
    #' (Variables x Thresholds x K).
    FDP_hat_array_BT = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      lst <- trex_da_get_fdp_hat_array_bt(private$ptr)
      if (length(lst) > 0) {
        simplify2array(lst)
      } else {
        lst
      }
    },

    #' @field Phi_array_BT A 3D array of variables selected for BT method.
    Phi_array_BT = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      lst <- trex_da_get_phi_array_bt(private$ptr)
      if (length(lst) > 0) {
        simplify2array(lst)
      } else {
        lst
      }
    },

    #' @field R_array_BT A 3D array mapping original to clustered features for BT method.
    R_array_BT = function(value) {
      if (!missing(value)) {
        stop("Field is read-only.")
      }
      lst <- trex_da_get_r_array_bt(private$ptr)
      if (length(lst) > 0) {
        simplify2array(lst)
      } else {
        lst
      }
    }
  )
)
