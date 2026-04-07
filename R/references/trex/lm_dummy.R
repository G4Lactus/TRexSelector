# ==============================================================================
# lm_dummy.R
#
# Performs one T-Rex random experiment: appends N(0,1) dummy predictors
# to X and runs a single T-LARS forward-selection path until T_stop dummies
# have entered the model.
#
# Public interface:
#   lm_dummy()           -- main entry point
#
# Internal helpers (not exported):
#   .check_lm_dummy()    -- input validation
#
# External dependencies (must be on the search path):
#   add_dummies(), tlars::tlars_model(), tlars::tlars(), tlars::tlars_cpp
# ==============================================================================


# ==============================================================================
# .check_lm_dummy
# ==============================================================================

.check_lm_dummy <- function(X, y, model_tlars, T_stop, num_dummies) {

  if (!is.matrix(X))              stop("'X' must be a matrix.")
  if (!is.numeric(X))             stop("'X' only allows numerical values.")
  if (anyNA(X))                   stop("'X' contains NAs.")
  if (!is.vector(drop(y)))        stop("'y' must be a vector.")
  if (!is.numeric(y))             stop("'y' only allows numerical values.")
  if (anyNA(y))                   stop("'y' contains NAs.")
  if (nrow(X) != length(drop(y))) stop("nrow(X) != length(y).")

  if (!missing(model_tlars) && !is.null(model_tlars)) {
    if (!methods::is(model_tlars, "tlars_cpp"))
      stop("'model_tlars' must be an object of class tlars_cpp.")
  }

  if (length(num_dummies) != 1 || num_dummies %% 1 != 0 || num_dummies < 1)
    stop("'num_dummies' must be a positive integer.")
  if (length(T_stop) != 1 || !(T_stop %in% seq(1, num_dummies)))
    stop(paste0("'T_stop' must be an integer in [1, ", num_dummies, "]."))

  invisible(NULL)
}


# ==============================================================================
# lm_dummy (main entry point)
# ==============================================================================

#' Perform one T-Rex random experiment
#'
#' Generates N(0,1) dummy predictors, appends them to X, and runs T-LARS
#' forward selection until T_stop dummies have been selected.
#'
#' A warm-start tlars_cpp object can be passed via \code{model_tlars} to
#' extend a previously computed path without restarting from scratch.
#'
#' @param X           Real valued predictor matrix (n x p).
#' @param y           Response vector (length n).
#' @param model_tlars tlars_cpp warm-start object (optional).
#' @param T_stop      Number of dummies before stopping. Default 1.
#' @param num_dummies Number of dummies appended to X. Default ncol(X).
#' @param type        'lar' (LARS, default) or 'lasso'.
#' @param early_stop  Logical. Stop after T_stop dummies. Default TRUE.
#' @param verbose     Logical. Show T-LARS progress. Default TRUE.
#' @param intercept   Logical. Include intercept. Default FALSE.
#' @param standardize Logical. Standardize X / centre y. Default TRUE.
#'
#' @return Object of class tlars_cpp.
#'
#' @importFrom methods is
#' @importFrom tlars tlars_model tlars tlars_cpp
#'
#' @export
lm_dummy <- function(X,
                     y,
                     model_tlars,
                     T_stop      = 1,
                     num_dummies = ncol(X),
                     type        = "lar",
                     early_stop  = TRUE,
                     verbose     = TRUE,
                     intercept   = FALSE,
                     standardize = TRUE) {

  # ── Validate and coerce inputs ─────────────────────────────────────────────

  type <- match.arg(type, c("lar", "lasso"))
  .check_lm_dummy(X, y,
                  if (missing(model_tlars)) NULL else model_tlars,
                  T_stop, num_dummies)

  # ── Build augmented matrix [X | dummies] on first call ─────────────────────

  first_call <- T_stop == 1 || missing(model_tlars) || is.null(model_tlars)

  if (first_call) {
    X_dummy <- add_dummies(X = X, num_dummies = num_dummies)

    model_tlars <- tlars::tlars_model(
      X           = X_dummy,
      y           = y,
      num_dummies = num_dummies,
      verbose     = FALSE,
      info        = FALSE,
      intercept   = intercept,
      standardize = standardize,
      type        = type
    )
  }

  # ── Execute T-LARS path step ──────────────────────────────────────────────

  tlars::tlars(
    model      = model_tlars,
    T_stop     = T_stop,
    early_stop = early_stop,
    info       = FALSE
  )

  return(model_tlars)
}
