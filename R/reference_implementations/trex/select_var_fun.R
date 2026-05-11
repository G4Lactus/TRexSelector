# ==============================================================================
# select_var_fun.R
#
# Selects the final variable set for the T-Rex selector by finding the
# (T*, v*) pair that maximises the number of selected variables while
# keeping the estimated FDP at or below the target level tFDR.
# doi:10.48550/arXiv.2110.06048
# ==============================================================================

#' Compute the set of selected variables
#'
#' Finds the optimal (T*, v*) that maximises the number of selected variables
#' subject to FDP_hat(T*, v*) <= tFDR, and returns the estimated support
#' vector for the T-Rex selector.
#'
#' @param p           Number of candidate variables.
#' @param tFDR        Target FDR level (in [0, 1]).
#' @param T_stop      Number of dummies at the stopping threshold.
#' @param FDP_hat_mat Conservative FDP estimates, one row per T value (T x |V|).
#' @param Phi_mat     Relative occurrences, one row per T value (T x p).
#' @param V           Voting level grid.
#'
#' @return Named list:
#'   \item{selected_var}{0/1 support vector of length p.}
#'   \item{v_thresh}{Optimal voting threshold v*.}
#'   \item{R_mat}{Number of selected variables matrix (T x |V|).}
#'
#' @export
select_var_fun <- function(p,
                           tFDR,
                           T_stop,
                           FDP_hat_mat,
                           Phi_mat,
                           V) {

  # Drop the last row at T_stop (path not yet terminated at that step)
  if (T_stop > 1) {
    FDP_hat_mat <- FDP_hat_mat[-T_stop, , drop = FALSE]
    Phi_mat     <- Phi_mat[-T_stop,     , drop = FALSE]
  }

  # Number of selected variables for each (T, v) pair
  R_mat <- matrix(NA_real_, nrow = nrow(FDP_hat_mat), ncol = ncol(FDP_hat_mat))
  for (TT in seq_len(nrow(FDP_hat_mat))) {
    for (VV in seq_along(V)) {
      R_mat[TT, VV] <- sum(Phi_mat[TT, ] > V[VV])
    }
  }

  # Maximise R over all (T, v) with FDP_hat <= tFDR
  FDP_hat_mat[FDP_hat_mat > tFDR] <- Inf
  val_max  <- suppressWarnings(max(R_mat[!is.infinite(FDP_hat_mat)]))
  ind_max  <- matrix(which(R_mat == val_max, arr.ind = TRUE), ncol = 2)

  # Tie-break: take the row with the largest v index (most conservative),
  # then the last such row
  ind_thresh <- ind_max[nrow(ind_max), ]

  v_thresh     <- V[ind_thresh[2]]
  selected_var <- rep(0L, times = p)
  selected_var[Phi_mat[ind_thresh[1], ] > v_thresh] <- 1L

  list(
    selected_var = selected_var,
    v_thresh     = v_thresh,
    R_mat        = R_mat
  )
}
