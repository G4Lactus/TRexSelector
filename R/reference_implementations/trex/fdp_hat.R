# ==============================================================================
# fdp_hat.R
#
# Conservative FDP estimator of the T-Rex selector evaluated at every
# voting level in the grid V.  doi:10.48550/arXiv.2110.06048
# ==============================================================================

#' Compute the conservative FDP estimate
#'
#' Evaluates the conservative FDP estimate of the T-Rex selector at
#' every voting level v in the grid V.
#'
#' @param V         Voting level grid (numeric vector in (0, 1]).
#' @param Phi       Relative occurrences for all variables (length-p vector).
#' @param Phi_prime Deflated relative occurrences (length-p vector).
#' @param eps       Numerical zero. Default .Machine$double.eps.
#'
#' @return Numeric vector of length(V) with conservative FDP estimates.
#'
#' @export
fdp_hat <- function(V,
                    Phi,
                    Phi_prime,
                    eps = .Machine$double.eps) {

  fdp_h <- rep(NA_real_, times = length(V))

  for (i in seq_along(V)) {
    num_sel_var <- sum(Phi > V[i])
    fdp_h[i]   <- if (num_sel_var < eps) {
      0
    } else {
      min(1, sum((1 - Phi_prime)[Phi > V[i]]) / num_sel_var)
    }
  }

  fdp_h
}
