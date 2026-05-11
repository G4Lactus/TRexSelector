# ==============================================================================
# Phi_prime_fun.R
#
# Computes the vector of deflated relative occurrences Phi' for all
# candidate variables j = 1, ..., p at stopping threshold T = T_stop.
# ==============================================================================

#' Compute deflated relative occurrences
#'
#' Returns the deflated relative occurrences Phi'[j] for all j = 1, ..., p
#' at T = T_stop, which are used by \code{fdp_hat()} to form the conservative
#' FDP estimate.
#'
#' @param p           Number of candidate variables.
#' @param T_stop      Number of dummies included at the stopping threshold.
#' @param num_dummies Total number of dummy predictors.
#' @param phi_T_mat   Matrix of relative occurrences (p x T_stop).
#' @param Phi         Relative occurrences at T = T_stop (length-p vector).
#' @param eps         Numerical zero. Default .Machine$double.eps.
#'
#' @return Length-p vector of deflated relative occurrences Phi'.
#'
#' @export
Phi_prime_fun <- function(p,
                          T_stop,
                          num_dummies,
                          phi_T_mat,
                          Phi,
                          eps = .Machine$double.eps) {

  av_num_var_sel  <- colSums(phi_T_mat)
  fifty_phi_T_mat <- phi_T_mat[Phi > 0.5, , drop = FALSE]
  delta_av        <- colSums(fifty_phi_T_mat)

  if (T_stop > 1) {

    # Convert cumulative sums to per-step increments
    delta_av[2:T_stop]              <- delta_av[2:T_stop]              - delta_av[1:(T_stop - 1)]
    phi_T_mat[, 2:T_stop]           <- phi_T_mat[, 2:T_stop]           - phi_T_mat[, 1:(T_stop - 1)]

    # Deflation scale per step
    phi_scale <- vapply(seq_along(delta_av), function(t) {
      if (delta_av[t] > eps) {
        1 - (((p - av_num_var_sel[t]) / (num_dummies - t + 1)) / delta_av[t])
      } else {
        0
      }
    }, numeric(1))

    Phi_prime <- phi_T_mat %*% phi_scale

  } else {
    Phi_prime <- if (delta_av[1] > eps) {
      (1 - (((p - av_num_var_sel[1]) / num_dummies) / delta_av[1])) * phi_T_mat[, 1]
    } else {
      rep(0, times = p)
    }
  }

  Phi_prime
}
