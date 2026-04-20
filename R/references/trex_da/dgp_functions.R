# ==============================================================================
# dgp_functions.R
#
# Data generating processes for each DA-TREX dependency scenario.
#
# Public functions:
#   dgp_ar1()     -- AR(1) Toeplitz covariance
#   dgp_equi()    -- Compound symmetry (single latent factor)
#   dgp_nn()      -- Banded covariance via MA(kappa) construction
#   dgp_bt()      -- Hierarchical block (two-level canonical BT scenario)
#   dgp_groups()  -- Multi-level latent factor (prior-groups scenario)
#
# All functions return:
#   list(X, y, beta, Sigma, n, p, s, snr)
#
# Shared model:  y = X %*% beta + eps,  eps ~ N(0, sigma^2 * I_n)
#   X rows i.i.d. N(0, Sigma),  Sigma has unit diagonal
#   beta is s-sparse with all active coefficients equal to `amplitude`
#
# Internal helpers (not exported):
#   .sim_beta()       -- sparse coefficient vector
#   .sim_response()   -- y = X %*% beta + noise
#   .snr()            -- signal-to-noise ratio
# ==============================================================================


# ==============================================================================
# Internal helpers
# ==============================================================================

#' Generate a sparse coefficient vector.
#'
#' Active variables occupy the first s positions by default, or are drawn
#' uniformly at random when random_support = TRUE.
#'
#' @param p         Number of predictors.
#' @param s         Number of active predictors (s < p).
#' @param amplitude Magnitude of each active coefficient. Default 3.
#' @param random_support If TRUE, active set is chosen uniformly at random.
#'                       Default FALSE.
#' @param seed      RNG seed. Default NULL.
#'
#' @return Numeric vector of length p.
.sim_beta <- function(p, s,
                       amplitude      = 3,
                       random_support = FALSE,
                       seed           = NULL) {
  if (!is.null(seed)) set.seed(seed)
  if (s >= p) stop("s must be strictly less than p.")

  beta <- rep(0, p)
  active <- if (random_support) sample(p, s, replace = FALSE) else seq(1, s)
  beta[active] <- amplitude
  beta
}


#' Generate the response vector.
#'
#' @param X       Predictor matrix (n x p).
#' @param beta    Coefficient vector (length p).
#' @param sigma   Noise standard deviation. Default 1.
#' @param seed    RNG seed. Default NULL.
#'
#' @return Numeric vector of length n.
.sim_response <- function(X, beta, sigma = 1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed + 99999L)
  drop(X %*% beta) + stats::rnorm(nrow(X), mean = 0, sd = sigma)
}


#' Compute the expected signal-to-noise ratio.
#'
#' SNR = beta^T Sigma beta / sigma^2  (variance of the signal per observation
#' divided by noise variance).
#'
#' @param beta  Coefficient vector (length p).
#' @param Sigma Covariance matrix (p x p).
#' @param sigma Noise standard deviation.
#'
#' @return Scalar SNR value.
.snr <- function(beta, Sigma, sigma) {
  drop(t(beta) %*% Sigma %*% beta) / sigma^2
}


# ==============================================================================
# dgp_ar1
# ==============================================================================

#' DGP for the AR(1) scenario (T-Rex+DA+AR1)
#'
#' Predictor rows follow a stationary AR(1) process along the variable index:
#'
#'   X[i, j] = rho * X[i, j-1] + sqrt(1 - rho^2) * eta[i,j]
#'
#' yielding the Toeplitz covariance Sigma[j,k] = rho^|j-k|.
#'
#' @param n         Number of observations.
#' @param p         Number of predictors.
#' @param rho       AR(1) autocorrelation coefficient. rho in (-1, 1).
#' @param s         Number of active predictors.
#' @param amplitude Active coefficient magnitude. Default 3.
#' @param sigma     Noise standard deviation. Default 1.
#' @param random_support If TRUE, active set chosen uniformly at random.
#' @param seed      RNG seed. Default NULL.
#'
#' @return List: X, y, beta, Sigma, n, p, s, snr.
#'
#' @examples
#' dat <- dgp_ar1(n = 100, p = 200, rho = 0.7, s = 5)
#' res <- trex_da(dat$X, dat$y, method = "trex+DA+AR1")
dgp_ar1 <- function(n, p, rho,
                    s              = 5,
                    amplitude      = 3,
                    sigma          = 1,
                    random_support = FALSE,
                    seed           = NULL) {

  if (!is.null(seed)) set.seed(seed)
  if (abs(rho) >= 1) stop("rho must be in (-1, 1) for a stationary AR(1).")

  # Generate X row-by-row via AR recursion (efficient; avoids p x p Cholesky)
  eta <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  X   <- matrix(0,                   nrow = n, ncol = p)

  X[, 1] <- eta[, 1]
  for (j in seq(2, p)) {
    X[, j] <- rho * X[, j - 1] + sqrt(1 - rho^2) * eta[, j]
  }

  # Toeplitz covariance: Sigma[j,k] = rho^|j-k|
  Sigma <- rho^abs(outer(seq(1, p), seq(1, p), `-`))

  beta <- .sim_beta(p, s, amplitude, random_support, seed)
  y    <- .sim_response(X, beta, sigma, seed)

  list(X = X, y = y, beta = beta, Sigma = Sigma,
       n = n, p = p, s = s, snr = .snr(beta, Sigma, sigma))
}


# ==============================================================================
# dgp_equi
# ==============================================================================

#' DGP for the equicorrelation scenario (T-Rex+DA+equi)
#'
#' Predictor rows follow a single latent factor model:
#'
#'   X[i, j] = sqrt(rho) * f[i] + sqrt(1 - rho) * eta[i, j]
#'
#' with f[i], eta[i,j] i.i.d. N(0,1), yielding compound symmetry:
#'
#'   Sigma[j,k] = rho  (j != k),   Sigma[j,j] = 1
#'
#' @param n         Number of observations.
#' @param p         Number of predictors.
#' @param rho       Equicorrelation coefficient.
#'                  rho in (-(p-1)^{-1}, 1) for positive definiteness.
#' @param s         Number of active predictors.
#' @param amplitude Active coefficient magnitude. Default 3.
#' @param sigma     Noise standard deviation. Default 1.
#' @param random_support If TRUE, active set chosen uniformly at random.
#' @param seed      RNG seed. Default NULL.
#'
#' @return List: X, y, beta, Sigma, n, p, s, snr.
#'
#' @examples
#' dat <- dgp_equi(n = 100, p = 200, rho = 0.5, s = 5)
#' res <- trex_da(dat$X, dat$y, method = "trex+DA+equi")
dgp_equi <- function(n, p, rho,
                     s              = 5,
                     amplitude      = 3,
                     sigma          = 1,
                     random_support = FALSE,
                     seed           = NULL) {

  if (!is.null(seed)) set.seed(seed)
  pd_lower <- -1 / (p - 1)
  if (rho <= pd_lower || rho >= 1)
    stop(paste0("rho must be in (", round(pd_lower, 4), ", 1) for p = ", p, "."))

  # Single latent factor construction (avoids storing p x p Sigma for generation)
  f   <- matrix(stats::rnorm(n),       nrow = n, ncol = 1)
  eta <- matrix(stats::rnorm(n * p),   nrow = n, ncol = p)
  X   <- sqrt(rho) * f %*% matrix(1, 1, p) + sqrt(1 - rho) * eta

  Sigma <- matrix(rho, nrow = p, ncol = p)
  diag(Sigma) <- 1

  beta <- .sim_beta(p, s, amplitude, random_support, seed)
  y    <- .sim_response(X, beta, sigma, seed)

  list(X = X, y = y, beta = beta, Sigma = Sigma,
       n = n, p = p, s = s, snr = .snr(beta, Sigma, sigma))
}


# ==============================================================================
# dgp_nn
# ==============================================================================

#' DGP for the nearest-neighbour scenario (T-Rex+DA+NN)
#'
#' Predictor rows follow an MA(kappa) process along the variable index:
#'
#'   X[i, j] = sum_{l=0}^{kappa} theta[l] * eta[i, j + l]
#'
#' where theta[l] = c * rho^l  (c chosen so Var(X[i,j]) = 1),
#' and eta[i, .] are i.i.d. N(0, 1).  This gives zero correlation exactly
#' for all lags |j - k| > kappa and a smooth decay within bandwidth:
#'
#'   Sigma[j,k] = rho^h * (sum_{l=0}^{kappa-h} rho^{2l}) /
#'                         (sum_{l=0}^{kappa}   rho^{2l}),   h = |j - k|
#'
#' @param n         Number of observations.
#' @param p         Number of predictors.
#' @param kappa     Bandwidth (max lag with non-zero correlation). Integer >= 1.
#' @param rho       Decay parameter within the bandwidth. rho in (0, 1).
#' @param s         Number of active predictors.
#' @param amplitude Active coefficient magnitude. Default 3.
#' @param sigma     Noise standard deviation. Default 1.
#' @param random_support If TRUE, active set chosen uniformly at random.
#' @param seed      RNG seed. Default NULL.
#'
#' @return List: X, y, beta, Sigma, n, p, s, snr, kappa.
#'
#' @examples
#' dat <- dgp_nn(n = 100, p = 200, kappa = 3, rho = 0.7, s = 5)
#' res <- trex_da(dat$X, dat$y, method = "trex+DA+NN")
dgp_nn <- function(n, p, kappa, rho,
                   s              = 5,
                   amplitude      = 3,
                   sigma          = 1,
                   random_support = FALSE,
                   seed           = NULL) {

  if (!is.null(seed)) set.seed(seed)
  if (kappa < 1 || kappa %% 1 != 0) stop("kappa must be a positive integer.")
  if (rho <= 0 || rho >= 1)         stop("rho must be in (0, 1).")

  # MA coefficients: theta[l] = rho^l, normalised to unit variance
  theta <- rho^seq(0, kappa)
  theta <- theta / sqrt(sum(theta^2))

  # Extended noise matrix: n x (p + kappa)
  eta <- matrix(stats::rnorm(n * (p + kappa)), nrow = n, ncol = p + kappa)

  # MA convolution: X[i, j] = eta[i, j:(j+kappa)] %*% theta
  X <- matrix(0, nrow = n, ncol = p)
  for (j in seq(1, p)) {
    X[, j] <- eta[, j:(j + kappa)] %*% theta
  }

  # Analytic banded covariance
  denom <- sum(theta^2)    # = 1 by construction
  Sigma <- matrix(0, nrow = p, ncol = p)
  for (h in seq(0, kappa)) {
    val     <- sum(theta[seq(1, kappa - h + 1)] * theta[seq(h + 1, kappa + 1)])
    indices <- which(abs(row(Sigma) - col(Sigma)) == h)
    Sigma[indices] <- val
  }

  beta <- .sim_beta(p, s, amplitude, random_support, seed)
  y    <- .sim_response(X, beta, sigma, seed)

  list(X = X, y = y, beta = beta, Sigma = Sigma,
       n = n, p = p, s = s, snr = .snr(beta, Sigma, sigma),
       kappa = kappa)
}


# ==============================================================================
# dgp_bt
# ==============================================================================

#' DGP for the binary tree / hierarchical clustering scenario (T-Rex+DA+BT)
#'
#' Two-level hierarchical block model.  The p variables are divided into
#' n_blocks equal-sized blocks.  Variables within the same block share a
#' within-block latent factor; all variables share a global factor:
#'
#'   X[i, j] = sqrt(rho_between)           * f0[i]
#'           + sqrt(rho_within - rho_between) * f_{m(j)}[i]
#'           + sqrt(1 - rho_within)          * eta[i, j]
#'
#' yielding:
#'
#'   Sigma[j,k] = rho_within   if same block
#'   Sigma[j,k] = rho_between  if different block
#'
#' @param n          Number of observations.
#' @param p          Number of predictors (must be divisible by n_blocks).
#' @param n_blocks   Number of top-level blocks.
#' @param rho_within Within-block correlation.
#' @param rho_between Between-block correlation. Default 0.
#'                    Must satisfy 0 <= rho_between < rho_within < 1.
#' @param s          Number of active predictors.
#' @param amplitude  Active coefficient magnitude. Default 3.
#' @param sigma      Noise standard deviation. Default 1.
#' @param random_support If TRUE, active set chosen uniformly at random.
#' @param seed       RNG seed. Default NULL.
#'
#' @return List: X, y, beta, Sigma, n, p, s, snr, block_labels.
#'
#' @examples
#' dat <- dgp_bt(n = 100, p = 200, n_blocks = 10,
#'               rho_within = 0.8, rho_between = 0.2, s = 5)
#' res <- trex_da(dat$X, dat$y, method = "trex+DA+BT")
dgp_bt <- function(n, p, n_blocks,
                   rho_within,
                   rho_between    = 0,
                   s              = 5,
                   amplitude      = 3,
                   sigma          = 1,
                   random_support = FALSE,
                   seed           = NULL) {

  if (!is.null(seed)) set.seed(seed)
  if (p %% n_blocks != 0)
    stop("p must be divisible by n_blocks.")
  if (rho_between < 0 || rho_between >= rho_within || rho_within >= 1)
    stop("Require 0 <= rho_between < rho_within < 1.")

  block_size   <- p / n_blocks
  block_labels <- rep(seq(1, n_blocks), each = block_size)

  # Global factor
  f0      <- stats::rnorm(n)
  # Per-block factors
  F_block <- matrix(stats::rnorm(n * n_blocks), nrow = n, ncol = n_blocks)
  # Idiosyncratic noise
  eta     <- matrix(stats::rnorm(n * p),        nrow = n, ncol = p)

  X <- sqrt(rho_between)              * matrix(f0, n, p) +
       sqrt(rho_within - rho_between) * F_block[, block_labels] +
       sqrt(1 - rho_within)           * eta

  # Block-structured covariance
  Sigma <- matrix(rho_between, nrow = p, ncol = p)
  for (m in seq(1, n_blocks)) {
    idx <- which(block_labels == m)
    Sigma[idx, idx] <- rho_within
  }
  diag(Sigma) <- 1

  beta <- .sim_beta(p, s, amplitude, random_support, seed)
  y    <- .sim_response(X, beta, sigma, seed)

  list(X = X, y = y, beta = beta, Sigma = Sigma,
       n = n, p = p, s = s, snr = .snr(beta, Sigma, sigma),
       block_labels = block_labels)
}


# ==============================================================================
# dgp_groups
# ==============================================================================

#' DGP for the prior-groups scenario (groups != NULL in trex_da)
#'
#' Multi-level latent factor model.  The user supplies a list of L group
#' assignments ordered \strong{coarse to fine} and an \strong{increasing}
#' vector of L correlation levels.  This matches the internal rho_grid
#' orientation of the BT and NN routes (x = 1 is coarsest, x = L finest).
#'
#' Each level x contributes a set of group-specific latent factors:
#'
#'   X[i, j] = sum_{x=1}^{L} sqrt(rho[x] - rho[x-1]) * f_{g_x(j)}[i]
#'           + sqrt(1 - rho[L]) * eta[i, j]
#'
#' with rho[0] = 0 by convention, so the loading at level x is
#' sqrt(rho_levels[x] - rho_levels[x-1]).
#'
#' @param n          Number of observations.
#' @param p          Number of predictors.
#' @param groups     List of L integer vectors of length p (group labels),
#'                   ordered \strong{coarse to fine}: groups[[1]] has the
#'                   fewest / largest clusters; groups[[L]] has the most /
#'                   smallest clusters.  Must satisfy the nesting property:
#'                   if j and k share a group at level x they must also share
#'                   a group at all coarser levels x' < x.
#' @param rho_levels \strong{Increasing} numeric vector of length L:
#'                   correlation at each level (coarse to fine).
#'                   All values in (0, 1), strictly increasing.
#'                   rho_levels[L] is the within-finest-group correlation.
#' @param s          Number of active predictors.
#' @param amplitude  Active coefficient magnitude. Default 3.
#' @param sigma      Noise standard deviation. Default 1.
#' @param random_support If TRUE, active set chosen uniformly at random.
#' @param seed       RNG seed. Default NULL.
#'
#' @return List: X, y, beta, Sigma, n, p, s, snr, groups, rho_levels.
#'
#' @examples
#' # Coarse to fine: level 1 = 2 big groups, level 3 = 8 singletons
#' groups <- list(
#'   c(1,1,1,1,2,2,2,2),   # level 1: coarse (rho ~ 0.2)
#'   c(1,1,2,2,3,3,4,4),   # level 2: medium (rho ~ 0.6)
#'   c(1,2,3,4,5,6,7,8)    # level 3: fine   (rho ~ 0.9)
#' )
#' dat <- dgp_groups(n = 100, p = 8, groups = groups,
#'                   rho_levels = c(0.2, 0.6, 0.9), s = 2)
#' res <- trex_da(dat$X, dat$y,
#'                groups          = groups,
#'                rho_grid_labels = c(0.2, 0.6, 0.9))
dgp_groups <- function(n, p, groups, rho_levels,
                       s              = 5,
                       amplitude      = 3,
                       sigma          = 1,
                       random_support = FALSE,
                       seed           = NULL) {

  if (!is.null(seed)) set.seed(seed)
  L <- length(groups)
  if (length(rho_levels) != L)
    stop("length(rho_levels) must equal length(groups).")
  if (!all(lengths(groups) == p))
    stop("Each element of groups must be a vector of length p.")
  if (any(diff(rho_levels) <= 0))
    stop("rho_levels must be strictly increasing (coarse to fine).")
  if (any(rho_levels <= 0) || any(rho_levels >= 1))
    stop("All rho_levels must be in (0, 1).")

  # rho_ext[1] = 0 (below coarsest), rho_ext[x+1] = rho_levels[x]
  # loading at level x = sqrt(rho_levels[x] - rho_levels[x-1])
  rho_ext <- c(0, rho_levels)

  X <- matrix(0, nrow = n, ncol = p)

  for (x in seq(1, L)) {
    loading   <- sqrt(rho_ext[x + 1] - rho_ext[x])   # positive since rho increasing
    group_ids <- unique(groups[[x]])

    for (gid in group_ids) {
      members      <- which(groups[[x]] == gid)
      f_x_g        <- stats::rnorm(n)
      X[, members] <- X[, members] + loading * f_x_g
    }
  }

  # Idiosyncratic noise uses finest-level variance: sqrt(1 - rho_levels[L])
  X <- X + sqrt(1 - rho_levels[L]) * matrix(stats::rnorm(n * p), n, p)

  # Analytic covariance: Sigma[j,k] = rho_levels[x*]
  # where x* = max level at which j and k share a group.
  # Scan from finest (x = L) down to coarsest (x = 1) and break on first match.
  Sigma <- matrix(0, nrow = p, ncol = p)
  for (j in seq(1, p)) {
    for (k in seq(j, p)) {
      shared_level <- 0
      for (x in seq(L, 1, by = -1)) {      # fine to coarse: first match = finest
        if (groups[[x]][j] == groups[[x]][k]) {
          shared_level <- x
          break
        }
      }
      Sigma[j, k] <- if (shared_level > 0) rho_levels[shared_level] else 0
      Sigma[k, j] <- Sigma[j, k]
    }
  }
  diag(Sigma) <- 1

  beta <- .sim_beta(p, s, amplitude, random_support, seed)
  y    <- .sim_response(X, beta, sigma, seed)

  list(X = X, y = y, beta = beta, Sigma = Sigma,
       n = n, p = p, s = s, snr = .snr(beta, Sigma, sigma),
       groups = groups, rho_levels = rho_levels)
}
