# =============================================================================
# dgp_correlated.R — Data Generating Processes for Correlated Predictors
#
# Provides two DGP functions that produce synthetic regression data with
# controlled predictor correlation structures, intended to test whether the
# DA variants of the Screen-T-Rex selector handle them correctly:
#
#   dgp_ar1()    – AR(1) column correlation  Cor(x_j, x_k) = rho^|j-k|
#   dgp_equi()   – Equicorrelation           Cor(x_j, x_k) = rho  (j != k)
#
# Both functions share the same:
#   - sparse coefficient vector beta (first p1 variables are active)
#   - response model  y = X %*% beta + epsilon
#   - noise level     sigma chosen to achieve a target SNR
#   - standardised X columns and centred y in the returned object
#
# Usage example
# -------------
#   source("dgp_correlated.R")
#   source("trex_screening.R")
#
#   set.seed(1)
#   d <- dgp_ar1(n = 300, p = 1000, p1 = 10, rho = 0.7, snr = 5)
#   res <- screen_trex(X = d$X, y = d$y, method = "trex+DA+AR1")
#   TPP(res$selected_var, d$beta)
#   FDP(res$selected_var, d$beta)
# =============================================================================


# =============================================================================
# Internal helpers
# =============================================================================

# -----------------------------------------------------------------------------
# .make_beta
#
# Constructs a sparse p-vector with p1 equally spaced active variables.
# Active coefficients are set to beta_val.  Active indices are evenly
# distributed across {1,...,p} so that the sparsity pattern is independent
# of the correlation structure.
#
# @param p        Total number of variables.
# @param p1       Number of active (non-zero) variables.
# @param beta_val Common value of the non-zero coefficients.
#
# @return Named list:  beta (length-p vector),  support (0/1 vector).
# -----------------------------------------------------------------------------
.make_beta <- function(p, p1, beta_val = 3) {
  beta    <- rep(0, times = p)
  # evenly spaced active indices to avoid all actives clustering in one
  # region of a correlation-structured X
  active  <- round(seq(1, p, length.out = p1 + 2L))[seq(2L, p1 + 1L)]
  beta[active] <- beta_val
  list(beta = beta, support = as.integer(beta != 0))
}


# -----------------------------------------------------------------------------
# .make_y
#
# Generates the response y = X %*% beta + epsilon, where sigma is chosen
# to achieve the requested signal-to-noise ratio:
#
#   SNR = Var(X %*% beta) / sigma^2 = (||X %*% beta||_2^2 / n) / sigma^2
#   =>  sigma = sqrt( ||X %*% beta||_2^2 / (n * SNR) )
#
# @param X    n x p design matrix (already standardised).
# @param beta Length-p coefficient vector.
# @param snr  Target signal-to-noise ratio.
#
# @return Named list:  y (centred response),  sigma (noise standard deviation).
# -----------------------------------------------------------------------------
.make_y <- function(X, beta, snr) {
  n      <- nrow(X)
  signal <- X %*% beta
  sigma  <- sqrt(sum(signal^2) / (n * snr))
  y      <- signal + stats::rnorm(n, mean = 0, sd = sigma)
  list(y = c(y - mean(y)), sigma = sigma)
}


# =============================================================================
# 1. dgp_ar1
# =============================================================================

#' Generate AR(1)-correlated regression data
#'
#' Produces a regression data set whose predictor matrix has an AR(1) column
#' correlation structure:
#'
#' \deqn{\mathrm{Cor}(x_j,\, x_k) = \rho^{|j-k|}, \quad \rho \in (-1,\,1).}
#'
#' The columns of \eqn{X} are generated recursively:
#' \deqn{x_1 \sim \mathcal{N}(0, I_n)}
#' \deqn{x_j = \rho \, x_{j-1} + \sqrt{1-\rho^2}\, w_j,\quad
#'       w_j \overset{\text{iid}}{\sim} \mathcal{N}(0, I_n),\quad j = 2,\ldots,p.}
#'
#' After generation all columns of \eqn{X} are standardised to unit variance.
#'
#' @param n        Number of observations.
#' @param p        Number of candidate variables.
#' @param p1       Number of active (non-zero) variables (default: 10).
#' @param rho      AR(1) autocorrelation parameter in (-1, 1) (default: 0.7).
#' @param snr      Signal-to-noise ratio (default: 5).
#' @param beta_val Common value of the non-zero regression coefficients
#'                 (default: 3).
#' @param seed     Integer RNG seed for reproducibility (default: NULL).
#'
#' @return A named list with components:
#' \describe{
#'   \item{\code{X}}{Standardised n x p design matrix.}
#'   \item{\code{y}}{Centred response vector of length n.}
#'   \item{\code{beta}}{True p-vector of regression coefficients.}
#'   \item{\code{support}}{Integer 0/1 support vector of length p.}
#'   \item{\code{active_set}}{Integer indices of the active variables.}
#'   \item{\code{sigma}}{Noise standard deviation used.}
#'   \item{\code{rho}}{AR(1) parameter used.}
#'   \item{\code{n, p, p1, snr}}{Input dimensions and SNR echoed.}
#' }
#'
#' @importFrom stats rnorm
#' @export
#'
#' @examples
#' set.seed(1)
#' d <- dgp_ar1(n = 300, p = 1000, p1 = 10, rho = 0.7, snr = 5)
#' cor(d$X[, 1:4])   # should show rho^|j-k| pattern
dgp_ar1 <- function(n,
                    p,
                    p1       = 10,
                    rho      = 0.7,
                    snr      = 5,
                    beta_val = 3,
                    seed     = NULL) {

  # --- Input validation ------------------------------------------------------
  stopifnot(
    is.numeric(n)   && length(n) == 1L   && n >= 2L,
    is.numeric(p)   && length(p) == 1L   && p >= 2L,
    is.numeric(p1)  && length(p1) == 1L  && p1 >= 1L && p1 < p,
    is.numeric(rho) && length(rho) == 1L && abs(rho) < 1,
    is.numeric(snr) && length(snr) == 1L && snr > 0
  )

  if (!is.null(seed)) set.seed(seed)

  # --- Predictor matrix with AR(1) column correlation -----------------------
  # Column-by-column recursion: x_j = rho * x_{j-1} + sqrt(1-rho^2) * w_j
  X        <- matrix(NA_real_, nrow = n, ncol = p)
  X[, 1L]  <- stats::rnorm(n)

  innovation_sd <- sqrt(1 - rho^2)
  for (j in seq(2L, p)) {
    X[, j] <- rho * X[, j - 1L] + innovation_sd * stats::rnorm(n)
  }

  # Standardise columns to unit variance (preserves the correlation structure)
  X <- scale(X)

  # --- Sparse regression coefficients ---------------------------------------
  bp      <- .make_beta(p, p1, beta_val)
  beta    <- bp$beta
  support <- bp$support

  # --- Response --------------------------------------------------------------
  yp    <- .make_y(X, beta, snr)
  y     <- yp$y
  sigma <- yp$sigma

  # --- Return ----------------------------------------------------------------
  list(
    X          = X,
    y          = y,
    beta       = beta,
    support    = support,
    active_set = which(support == 1L),
    sigma      = sigma,
    rho        = rho,
    n          = n,
    p          = p,
    p1         = p1,
    snr        = snr
  )
}


# =============================================================================
# 2. dgp_equi
# =============================================================================

#' Generate equicorrelated regression data
#'
#' Produces a regression data set whose predictor matrix has an equicorrelation
#' structure: all pairs of distinct columns share the same correlation \eqn{\rho}:
#'
#' \deqn{\mathrm{Cor}(x_j,\, x_k) = \rho, \quad j \neq k,\quad \rho \in [0,\,1).}
#'
#' The columns of \eqn{X} are generated via a one-factor model:
#' \deqn{x_j = \sqrt{\rho}\, z + \sqrt{1-\rho}\, w_j,}
#' where \eqn{z \sim \mathcal{N}(0, I_n)} is the common factor shared by all
#' columns and \eqn{w_j \overset{\text{iid}}{\sim} \mathcal{N}(0, I_n)} are
#' idiosyncratic noise columns.
#'
#' After generation all columns of \eqn{X} are standardised to unit variance.
#'
#' @param n        Number of observations.
#' @param p        Number of candidate variables.
#' @param p1       Number of active (non-zero) variables (default: 10).
#' @param rho      Equicorrelation parameter in [0, 1) (default: 0.7).
#' @param snr      Signal-to-noise ratio (default: 5).
#' @param beta_val Common value of the non-zero regression coefficients
#'                 (default: 3).
#' @param seed     Integer RNG seed for reproducibility (default: NULL).
#'
#' @return A named list with the same components as \code{dgp_ar1()}.
#'
#' @importFrom stats rnorm
#' @export
#'
#' @examples
#' set.seed(1)
#' d <- dgp_equi(n = 300, p = 1000, p1 = 10, rho = 0.7, snr = 5)
#' round(cor(d$X[, 1:4]), 2)   # should be approximately rho off-diagonal
dgp_equi <- function(n,
                     p,
                     p1       = 10,
                     rho      = 0.7,
                     snr      = 5,
                     beta_val = 3,
                     seed     = NULL) {

  # --- Input validation ------------------------------------------------------
  stopifnot(
    is.numeric(n)   && length(n) == 1L   && n >= 2L,
    is.numeric(p)   && length(p) == 1L   && p >= 2L,
    is.numeric(p1)  && length(p1) == 1L  && p1 >= 1L && p1 < p,
    is.numeric(rho) && length(rho) == 1L && rho >= 0 && rho < 1,
    is.numeric(snr) && length(snr) == 1L && snr > 0
  )

  if (!is.null(seed)) set.seed(seed)

  # --- Predictor matrix with equicorrelation --------------------------------
  # One-factor model:  x_j = sqrt(rho) * z + sqrt(1-rho) * w_j
  z  <- stats::rnorm(n)                              # shared common factor
  W  <- matrix(stats::rnorm(n * p), nrow = n)        # idiosyncratic noise

  X  <- sqrt(rho) * matrix(z, nrow = n, ncol = p, byrow = FALSE) +
    sqrt(1 - rho) * W

  # Standardise columns
  X <- scale(X)

  # --- Sparse regression coefficients ---------------------------------------
  bp      <- .make_beta(p, p1, beta_val)
  beta    <- bp$beta
  support <- bp$support

  # --- Response --------------------------------------------------------------
  yp    <- .make_y(X, beta, snr)
  y     <- yp$y
  sigma <- yp$sigma

  # --- Return ----------------------------------------------------------------
  list(
    X          = X,
    y          = y,
    beta       = beta,
    support    = support,
    active_set = which(support == 1L),
    sigma      = sigma,
    rho        = rho,
    n          = n,
    p          = p,
    p1         = p1,
    snr        = snr
  )
}

# =============================================================================
# 3. Block equicorrelation DGP
# =============================================================================

#' Generate block-equicorrelated regression data
#'
#' Produces a regression data set whose predictor matrix has a block
#' equicorrelation structure. The `p` variables are divided into `n_blocks`
#' independent blocks. Within each block, pairs of distinct columns share the
#' same correlation `rho`. Between blocks, columns are strictly uncorrelated.
#'
#' The columns of X are generated via a block-specific factor model:
#' x_j = sqrt(rho) * z_{0k} + sqrt(1-rho) * w_j,  for j in block k
#' where z_{0k} ~ N(0, I_n) is the block's common factor, and w_j are
#' idiosyncratic noise columns.
#'
#' @param n Number of observations.
#' @param p Number of candidate variables.
#' @param p1 Number of active (non-zero) variables (default 10).
#' @param rho Equicorrelation parameter within blocks in [0, 1) (default 0.7).
#' @param n_blocks Number of independent blocks (default 5).
#' @param snr Signal-to-noise ratio (default 5).
#' @param betaval Common value of the non-zero regression coefficients (default 3).
#' @param seed Integer RNG seed for reproducibility (default NULL).
#'
#' @return A named list with components similar to `dgp_equi` and `dgp_ar1`,
#'         along with the additional `n_blocks` integer.
#' @importFrom stats rnorm
#' @export
dgp_block_equi <- function(n, p, p1 = 10, rho = 0.7, n_blocks = 5, snr = 5, betaval = 3, seed = NULL) {

  # --- Input validation ------------------------------------------------------
  stopifnot(
    is.numeric(n) && length(n) == 1L && n >= 2L,
    is.numeric(p) && length(p) == 1L && p >= 2L,
    is.numeric(p1) && length(p1) == 1L && p1 >= 1L && p1 <= p,
    is.numeric(rho) && length(rho) == 1L && rho >= 0 && rho < 1,
    is.numeric(n_blocks) && length(n_blocks) == 1L && n_blocks >= 1L && n_blocks <= p,
    is.numeric(snr) && length(snr) == 1L && snr > 0
  )
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # --- Block structure -------------------------------------------------------
  # Determine block sizes (handling cases where p is not perfectly divisible)
  base_size <- p %/% n_blocks
  remainder <- p %% n_blocks
  block_sizes <- rep(base_size, n_blocks)
  if (remainder > 0) {
    block_sizes[1:remainder] <- block_sizes[1:remainder] + 1
  }

  # --- Predictor Matrix Generation -------------------------------------------
  X <- matrix(NA_real_, nrow = n, ncol = p)

  col_idx <- 1
  for (k in seq_len(n_blocks)) {
    bk_size <- block_sizes[k]

    # Shared common factor for the k-th block z_{0k}
    z_k <- stats::rnorm(n)

    # Idiosyncratic noise for the variables in the k-th block
    W_k <- matrix(stats::rnorm(n * bk_size), nrow = n, ncol = bk_size)

    # Generate block columns: xj = sqrt(rho)*z_k + sqrt(1-rho)*w_j
    X_k <- sqrt(rho) * matrix(z_k, nrow = n, ncol = bk_size, byrow = FALSE) + sqrt(1 - rho) * W_k

    # Assign to main matrix
    idx_range <- col_idx:(col_idx + bk_size - 1)
    X[, idx_range] <- X_k
    col_idx <- col_idx + bk_size
  }

  # Standardise columns to unit variance (preserves the correlation structure)
  X <- scale(X)

  # --- Sparse regression coefficients ---------------------------------------
  bp <- .make_beta(p, p1, betaval)
  beta <- bp$beta
  support <- bp$support

  # --- Response --------------------------------------------------------------
  yp <- .make_y(X, beta, snr)
  y <- yp$y
  sigma <- yp$sigma

  # --- Return ----------------------------------------------------------------
  list(
    X = X,
    y = y,
    beta = beta,
    support = support,
    activeset = which(support == 1L),
    sigma = sigma,
    rho = rho,
    n_blocks = n_blocks,
    n = n,
    p = p,
    p1 = p1,
    snr = snr
  )
}


# =============================================================================
# 4. Diagnostics
# =============================================================================

#' Empirical correlation diagnostics for a DGP object
#'
#' Computes a small set of scalar summaries to verify that the generated
#' design matrix has the intended correlation structure.
#'
#' @param dgp  Output of \code{dgp_ar1()}, \code{dgp_equi()}, or \code{dgp_block_equi()}.
#' @param type \code{"ar1"}, \code{"equi"}, or \code{"block_equi"}.
#' @param max_col Maximum number of columns used for the full correlation
#'                matrix (guards against large p); default 50.
#'
#' @return A named list:
#' \describe{
#'   \item{\code{rho_theoretical}}{The rho value used during generation.}
#'   \item{\code{mean_off_diag_cor}}{Mean off-diagonal empirical correlation
#'         of the first \code{max_col} columns.}
#'   \item{\code{lag1_cor}}{(AR1 only) Mean empirical lag-1 column correlation.}
#'   \item{\code{lag2_cor}}{(AR1 only) Mean empirical lag-2 column correlation.}
#'   \item{\code{expected_lag1}}{(AR1 only) Theoretical lag-1 = rho.}
#'   \item{\code{expected_lag2}}{(AR1 only) Theoretical lag-2 = rho^2.}
#' }
#' @export
dgp_diagnostics <- function(dgp, type = c("ar1", "equi", "block_equi"), max_col = 50L) {
  type <- match.arg(type)
  X    <- dgp$X[, seq_len(min(ncol(dgp$X), max_col)), drop = FALSE]
  Cor  <- stats::cor(X)

  # Off-diagonal mean
  off_diag_idx     <- which(lower.tri(Cor))
  mean_off_diag    <- mean(Cor[off_diag_idx])

  out <- list(
    rho_theoretical  = dgp$rho,
    mean_off_diag_cor = round(mean_off_diag, 4L)
  )

  if (type == "ar1") {
    nc <- ncol(X)
    lag1_vals <- vapply(seq_len(nc - 1L),
                        function(j) Cor[j, j + 1L], numeric(1L))
    lag2_vals <- if (nc >= 3L) {
      vapply(seq_len(nc - 2L), function(j) Cor[j, j + 2L], numeric(1L))
    } else {
      NA_real_
    }
    out$lag1_cor      <- round(mean(lag1_vals), 4L)
    out$lag2_cor      <- round(mean(lag2_vals), 4L)
    out$expected_lag1 <- dgp$rho
    out$expected_lag2 <- dgp$rho^2
  }

  out
}
