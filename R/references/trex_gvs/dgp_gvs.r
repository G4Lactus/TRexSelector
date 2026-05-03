# ==============================================================================
# dgp_gvs.R
#
# Data generating processes (DGPs) for benchmarking the T-Rex+GVS selector.
#
# All DGPs follow the linear regression model:
#
#   y = X * beta + sigma_eps * epsilon,   epsilon ~ N(0, I_n)
#
# where X in R^{n x p} has a block-group covariance structure that the GVS
# selector is designed to exploit.
# DGPs differ in:
#   - the within-group correlation structure of X,
#   - the between-group correlation (null case vs. stress test),
#   - the distribution of group sizes, and
#   - the placement of active variables inside groups.
#
# All functions return a named list with identical fields (see .dgp_output_gvs())
# so that benchmark code can operate on a uniform interface.
#
# Public interface:
#   dgp_block_equi()     -- DGP 1: block equicorrelation, equal group sizes
#   dgp_block_ar1()      -- DGP 2: AR(1) within groups, equal group sizes
#   dgp_block_unequal()  -- DGP 3: block equicorrelation, unequal group sizes
#   dgp_between_corr()   -- DGP 4: block equicorrelation + weak between-group
#                                  correlation (corr_max stress test)
#
# Internal helpers (.):
#   .check_dgp_gvs()     -- shared input validation
#   .block_equi_cov()    -- block equicorrelation covariance matrix
#   .block_ar1_cov()     -- AR(1) within-group covariance matrix
#   .draw_X()            -- draw X from N(0, Sigma) via Cholesky
#   .make_active()       -- place active variables inside groups
#   .dgp_output_gvs()    -- assemble consistent return list
# ==============================================================================


# ==============================================================================
# .check_dgp_gvs
# ==============================================================================

#' Shared input validation for all GVS DGPs.
#'
#' @param n       Sample size.
#' @param p       Total number of variables.
#' @param M       Number of groups (clusters). Must divide p evenly for
#'                equal-size DGPs; relaxed for dgp_block_unequal().
#' @param s       Number of active variables per group. Must satisfy s <= b
#'                where b is the group size.
#' @param rho_in  Within-group correlation parameter.
#' @param snr     Signal-to-noise ratio E[||X beta||^2] / (n * sigma_eps^2).
#' @param sigma_eps Noise standard deviation.
#' @param b_min   Minimum group size (used in unequal-size check). Default 1.
.check_dgp_gvs <- function(n, p, M, s, rho_in, snr, sigma_eps, b_min = 1L) {

  if (length(n) != 1 || n %% 1 != 0 || n < 2)
    stop("'n' must be an integer >= 2.")
  if (length(p) != 1 || p %% 1 != 0 || p < 1)
    stop("'p' must be a positive integer.")
  if (length(M) != 1 || M %% 1 != 0 || M < 1 || M > p)
    stop(paste0("'M' must be an integer in [1, p] = [1, ", p, "]."))
  if (length(s) != 1 || s %% 1 != 0 || s < 1 || s > b_min)
    stop(paste0("'s' must be an integer in [1, min_group_size] = [1, ", b_min, "]."))
  if (length(rho_in) != 1 || rho_in < 0 || rho_in >= 1)
    stop("'rho_in' must be in [0, 1).")
  if (length(snr) != 1 || snr <= 0)
    stop("'snr' must be a positive scalar.")
  if (length(sigma_eps) != 1 || sigma_eps <= 0)
    stop("'sigma_eps' must be a positive scalar.")

  invisible(NULL)
}


# ==============================================================================
# .block_equi_cov
# ==============================================================================

#' Build a block equicorrelation covariance matrix for p variables in M groups.
#'
#' Within group m (variables in groups_list[[m]]):
#'   Sigma_{jj'} = rho_in  for j != j'
#'   Sigma_{jj}  = 1       (marginal variance normalised to 1)
#'
#' Between groups m and m' (j in G_m, j' in G_m'):
#'   Sigma_{jj'} = rho_out
#'
#' The matrix is positive definite iff
#'   rho_in  > -1 / (max_m p_m - 1)   and
#'   rho_out < rho_in   (or rho_out small enough for the full block).
#'
#' @param p           Total number of variables.
#' @param groups_list List of M integer index vectors (partition of 1:p).
#' @param rho_in      Within-group equicorrelation in [0, 1).
#' @param rho_out     Between-group correlation in [0, 1). Default 0.
#'
#' @return p x p symmetric positive definite covariance matrix.
.block_equi_cov <- function(p, groups_list, rho_in, rho_out = 0) {

  Sigma <- matrix(rho_out, nrow = p, ncol = p)
  diag(Sigma) <- 1

  for (idx in groups_list) {
    Sigma[idx, idx]        <- rho_in
    Sigma[cbind(idx, idx)] <- 1        # restore unit diagonal
  }

  # Numerical PD check
  ev_min <- min(eigen(Sigma, only.values = TRUE)$values)
  if (ev_min <= 0)
    stop(sprintf(
      "Covariance matrix is not positive definite (lambda_min = %.4g). ",
      ev_min,
      "Reduce rho_in, rho_out, or the group sizes."
    ))

  Sigma
}


# ==============================================================================
# .block_ar1_cov
# ==============================================================================

#' Build a block AR(1) covariance matrix for p variables in M groups.
#'
#' Within group m (variables in groups_list[[m]], ordered by index):
#'   Sigma_{jj'} = rho_in^|rank(j) - rank(j')|
#'
#' Between groups: Sigma_{jj'} = 0.
#'
#' @param p           Total number of variables.
#' @param groups_list List of M integer index vectors (partition of 1:p).
#' @param rho_in      AR(1) parameter in [0, 1).
#'
#' @return p x p symmetric positive definite covariance matrix.
.block_ar1_cov <- function(p, groups_list, rho_in) {

  Sigma <- diag(p)

  for (idx in groups_list) {
    b_m <- length(idx)
    ranks <- seq(b_m)
    Sigma[idx, idx] <- outer(ranks, ranks, function(i, j) rho_in^abs(i - j))
  }

  Sigma
}


# ==============================================================================
# .draw_X
# ==============================================================================

#' Draw X in R^{n x p} with rows i.i.d. N(0, Sigma).
#'
#' Uses the lower Cholesky factor: X = W L', W ~ N(0, I).
#' Column-centres the result to remove finite-sample mean shifts.
#'
#' @param n     Sample size.
#' @param Sigma p x p positive definite covariance matrix.
#'
#' @return n x p column-centred matrix.
.draw_X <- function(n, Sigma) {
  p <- ncol(Sigma)
  L <- t(chol(Sigma))                                  # lower Cholesky
  W <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  X <- W %*% t(L)
  X - matrix(colMeans(X), nrow = n, ncol = p, byrow = TRUE)
}


# ==============================================================================
# .make_active
# ==============================================================================

#' Select s active variables within each group and set their coefficients.
#'
#' Active variables are the FIRST s indices of each group (sorted). This is
#' deterministic conditional on the group structure and makes the ground truth
#' unambiguous.
#'
#' The coefficient magnitude is chosen to achieve the target SNR:
#'
#'   SNR = E[||X beta||^2] / (n sigma_eps^2)
#'       = beta' Sigma beta / sigma_eps^2
#'
#' Under equicorrelation within groups and between-group independence, and
#' with a constant coefficient c on all active variables:
#'
#'   beta' Sigma beta = c^2 * sum_m ( s + s*(s-1)*rho_in )
#'                    = c^2 * s * M * (1 + (s-1)*rho_in)
#'
#' Solving for c: c = sigma_eps * sqrt(SNR / (s * M * (1 + (s-1)*rho_in))).
#'
#' For non-equicorrelation DGPs (AR(1)), we use Sigma directly.
#'
#' @param p           Total number of variables.
#' @param groups_list List of M integer index vectors.
#' @param s           Number of active variables per group.
#' @param Sigma       p x p population covariance of X (used for exact SNR).
#' @param snr         Target signal-to-noise ratio.
#' @param sigma_eps   Noise standard deviation.
#'
#' @return Named list:
#'   active_set  — integer vector of all active variable indices
#'   beta        — length-p coefficient vector (zero at null variables)
#'   beta_coef   — scalar coefficient magnitude c
.make_active <- function(p, groups_list, s, Sigma, snr, sigma_eps) {

  M          <- length(groups_list)
  active_set <- unlist(lapply(groups_list, function(idx) sort(idx)[seq(s)]))
  beta       <- rep(0, times = p)

  # Coefficient magnitude calibrated to target SNR
  beta_sub  <- rep(1, times = length(active_set))
  quad_form <- as.numeric(t(beta_sub) %*% Sigma[active_set, active_set] %*% beta_sub)
  c         <- sigma_eps * sqrt(snr / quad_form)

  beta[active_set] <- c

  list(
    active_set = active_set,
    beta       = beta,
    beta_coef  = c
  )
}


# ==============================================================================
# .dgp_output_gvs
# ==============================================================================

#' Assemble the standard GVS DGP return list.
#'
#' All DGP functions return this identical structure so that downstream
#' benchmark code (FDR / TPR evaluation, visualisation) can operate on a
#' uniform interface.
#'
#' @param X           n x p column-centred predictor matrix.
#' @param y           Length-n response vector.
#' @param beta        Length-p true coefficient vector.
#' @param groups_list List of M integer index vectors (true group partition).
#' @param active_set  Integer vector of truly active variable indices.
#' @param Sigma       p x p population covariance of X.
#' @param sigma_eps   Noise standard deviation.
#' @param params      Named list of all DGP parameters (for reproducibility).
#'
#' @return Named list with fields:
#'   X            — n x p predictor matrix (column-centred, unscaled)
#'   y            — length-n response
#'   beta         — length-p true coefficient vector
#'   groups_list  — list of M integer index vectors (true clusters)
#'   groups       — integer vector of length p (groups[j] = cluster index of j)
#'   active_set   — integer vector of active variable indices
#'   null_set     — integer vector of null variable indices
#'   M            — number of groups
#'   group_sizes  — integer vector of per-group sizes (p_1,...,p_M)
#'   Sigma        — population covariance of X
#'   sigma_eps    — noise standard deviation
#'   SNR          — achieved population SNR (beta' Sigma beta / sigma_eps^2)
#'   params       — full parameter list for reproducibility
.dgp_output_gvs <- function(X, y, beta, groups_list, active_set,
                              Sigma, sigma_eps, params) {

  p          <- ncol(X)
  M          <- length(groups_list)
  null_set   <- setdiff(seq(p), active_set)
  group_sizes <- lengths(groups_list)

  # groups vector: groups[j] = cluster index of variable j
  groups <- integer(p)
  for (m in seq(M)) groups[groups_list[[m]]] <- m

  # Population SNR
  SNR <- as.numeric(t(beta) %*% Sigma %*% beta) / sigma_eps^2

  list(
    X           = X,
    y           = y,
    beta        = beta,
    groups_list = groups_list,
    groups      = groups,
    active_set  = active_set,
    null_set    = null_set,
    M           = M,
    group_sizes = group_sizes,
    Sigma       = Sigma,
    sigma_eps   = sigma_eps,
    SNR         = SNR,
    params      = params
  )
}


# ==============================================================================
# dgp_block_equi  (DGP 1)
# ==============================================================================

#' DGP 1: Block equicorrelation, equal group sizes.
#'
#' The canonical GVS test case (Machkour et al., CAMSAP 2023, Section IV-A).
#' Variables are partitioned into M disjoint groups of equal size b = p / M.
#' Within each group, the population correlation is the constant rho_in;
#' between groups the correlation is zero. The first s variables of each
#' group are active with equal coefficient magnitude calibrated to snr.
#'
#' Population covariance of X:
#'
#'   Sigma_{jj'}  = 1       if j = j'
#'   Sigma_{jj'}  = rho_in  if j != j' and same group
#'   Sigma_{jj'}  = 0       otherwise
#'
#' @param n       Sample size.
#' @param p       Total number of variables. Must be divisible by M.
#' @param M       Number of groups. Default 10.
#' @param s       Number of active variables per group. Default 1.
#' @param rho_in  Within-group equicorrelation in [0, 1). Default 0.7.
#' @param snr     Target population SNR. Default 5.
#' @param sigma_eps Noise standard deviation. Default 1.
#' @param seed    RNG seed. Default NULL.
#'
#' @return Standard GVS DGP list (see .dgp_output_gvs()).
#'
#' @examples
#' dgp <- dgp_block_equi(n = 100, p = 200, M = 10, s = 1, rho_in = 0.7, seed = 1)
#' dim(dgp$X)       # 100 x 200
#' dgp$group_sizes  # rep(20, 10)
#' dgp$SNR
dgp_block_equi <- function(n, p, M = 10, s = 1, rho_in = 0.7,
                             snr = 5, sigma_eps = 1, seed = NULL) {

  if (p %% M != 0)
    stop(paste0("'p' must be divisible by 'M'. Got p = ", p, ", M = ", M, "."))

  b <- p %/% M
  .check_dgp_gvs(n, p, M, s, rho_in, snr, sigma_eps, b_min = b)
  if (!is.null(seed)) set.seed(seed)

  groups_list <- lapply(seq(M), function(m) ((m - 1L) * b + 1L):(m * b))
  Sigma       <- .block_equi_cov(p, groups_list, rho_in, rho_out = 0)
  X           <- .draw_X(n, Sigma)
  act         <- .make_active(p, groups_list, s, Sigma, snr, sigma_eps)
  eps         <- stats::rnorm(n)
  y           <- X %*% act$beta + sigma_eps * eps

  .dgp_output_gvs(X, y, act$beta, groups_list, act$active_set, Sigma,
                   sigma_eps,
                   params = list(n = n, p = p, M = M, s = s,
                                 rho_in = rho_in, snr = snr,
                                 sigma_eps = sigma_eps, seed = seed))
}


# ==============================================================================
# dgp_block_ar1  (DGP 2)
# ==============================================================================

#' DGP 2: AR(1) within-group correlation, equal group sizes.
#'
#' Identical group structure to dgp_block_equi() but within each group the
#' population correlation between variables at ranks r and r' is rho_in^|r-r'|
#' (Toeplitz). Between groups the correlation is zero. This tests whether
#' the IEN penalty's use of the binary support vector 1_m is robust when
#' within-group correlations are non-uniform.
#'
#' Population covariance of X:
#'
#'   Sigma_{jj'} = rho_in^|rank(j,Gm) - rank(j',Gm)|  if same group Gm
#'   Sigma_{jj'} = 0                                    if different groups
#'   Sigma_{jj}  = 1
#'
#' @param n       Sample size.
#' @param p       Total number of variables. Must be divisible by M.
#' @param M       Number of groups. Default 10.
#' @param s       Number of active variables per group. Default 1.
#' @param rho_in  AR(1) within-group parameter in [0, 1). Default 0.7.
#' @param snr     Target population SNR. Default 5.
#' @param sigma_eps Noise standard deviation. Default 1.
#' @param seed    RNG seed. Default NULL.
#'
#' @return Standard GVS DGP list (see .dgp_output_gvs()).
#'
#' @examples
#' dgp <- dgp_block_ar1(n = 100, p = 200, M = 10, s = 1, rho_in = 0.7, seed = 1)
dgp_block_ar1 <- function(n, p, M = 10, s = 1, rho_in = 0.7,
                            snr = 5, sigma_eps = 1, seed = NULL) {

  if (p %% M != 0)
    stop(paste0("'p' must be divisible by 'M'. Got p = ", p, ", M = ", M, "."))

  b <- p %/% M
  .check_dgp_gvs(n, p, M, s, rho_in, snr, sigma_eps, b_min = b)
  if (!is.null(seed)) set.seed(seed)

  groups_list <- lapply(seq(M), function(m) ((m - 1L) * b + 1L):(m * b))
  Sigma       <- .block_ar1_cov(p, groups_list, rho_in)
  X           <- .draw_X(n, Sigma)
  act         <- .make_active(p, groups_list, s, Sigma, snr, sigma_eps)
  eps         <- stats::rnorm(n)
  y           <- X %*% act$beta + sigma_eps * eps

  .dgp_output_gvs(X, y, act$beta, groups_list, act$active_set, Sigma,
                   sigma_eps,
                   params = list(n = n, p = p, M = M, s = s,
                                 rho_in = rho_in, snr = snr,
                                 sigma_eps = sigma_eps, seed = seed))
}


# ==============================================================================
# dgp_block_unequal  (DGP 3)
# ==============================================================================

#' DGP 3: Block equicorrelation, geometrically distributed group sizes.
#'
#' Group sizes p_1,...,p_M follow a geometric progression:
#'
#'   p_m = b_min * r^(m-1),  m = 1,...,M
#'
#' where r > 1 is the growth ratio and the sizes are rounded and rescaled
#' so that sum(p_m) = p exactly. This tests the 1/sqrt(p_m) normalisation
#' of the IEN bottom block: large groups must not dominate the penalty.
#'
#' @param n       Sample size.
#' @param p       Total number of variables.
#' @param M       Number of groups. Default 6.
#' @param s       Number of active variables per group (applied to every
#'                group; must satisfy s <= min(p_m)). Default 1.
#' @param rho_in  Within-group equicorrelation in [0, 1). Default 0.7.
#' @param ratio   Geometric growth ratio r > 1 for group sizes. Default 2.
#' @param snr     Target population SNR. Default 5.
#' @param sigma_eps Noise standard deviation. Default 1.
#' @param seed    RNG seed. Default NULL.
#'
#' @return Standard GVS DGP list (see .dgp_output_gvs()), with
#'   params$ratio added.
#'
#' @examples
#' dgp <- dgp_block_unequal(n = 100, p = 126, M = 6, s = 1, ratio = 2, seed = 1)
#' dgp$group_sizes   # approximate powers of 2
dgp_block_unequal <- function(n, p, M = 6, s = 1, rho_in = 0.7,
                               ratio = 2, snr = 5, sigma_eps = 1,
                               seed = NULL) {

  if (length(ratio) != 1 || ratio <= 1)
    stop("'ratio' must be a scalar > 1.")

  # Build geometrically spaced sizes and rescale to sum to p
  raw   <- ratio^(seq(0, M - 1))
  sizes <- as.integer(round(raw / sum(raw) * p))

  # Correct rounding: distribute residual to the last group
  residual <- p - sum(sizes)
  sizes[M] <- sizes[M] + residual

  b_min <- min(sizes)
  if (b_min < 1)
    stop("Smallest group size is 0. Increase 'p' or reduce 'M' or 'ratio'.")

  .check_dgp_gvs(n, p, M, s, rho_in, snr, sigma_eps, b_min = b_min)
  if (!is.null(seed)) set.seed(seed)

  # Build groups: place them consecutively
  ends        <- cumsum(sizes)
  starts      <- c(1L, ends[-M] + 1L)
  groups_list <- lapply(seq(M), function(m) starts[m]:ends[m])

  Sigma <- .block_equi_cov(p, groups_list, rho_in, rho_out = 0)
  X     <- .draw_X(n, Sigma)
  act   <- .make_active(p, groups_list, s, Sigma, snr, sigma_eps)
  eps   <- stats::rnorm(n)
  y     <- X %*% act$beta + sigma_eps * eps

  .dgp_output_gvs(X, y, act$beta, groups_list, act$active_set, Sigma,
                   sigma_eps,
                   params = list(n = n, p = p, M = M, s = s,
                                 rho_in = rho_in, ratio = ratio,
                                 snr = snr, sigma_eps = sigma_eps,
                                 seed = seed))
}


# ==============================================================================
# dgp_between_corr  (DGP 4)
# ==============================================================================

#' DGP 4: Block equicorrelation with weak between-group correlation.
#'
#' Identical to dgp_block_equi() but adds a non-zero between-group
#' correlation rho_out > 0. When rho_out approaches corr_max, the
#' hierarchical clustering algorithm may merge groups incorrectly.
#' This is the primary stress test for the corr_max threshold and
#' the hc_method choice in trex_GVS().
#'
#' Population covariance of X:
#'
#'   Sigma_{jj'}  = 1       if j = j'
#'   Sigma_{jj'}  = rho_in  if j != j' and same group
#'   Sigma_{jj'}  = rho_out if different groups
#'
#' For the matrix to be positive definite we need rho_out < rho_in
#' (otherwise within-group and between-group correlations are
#' indistinguishable).
#'
#' @param n        Sample size.
#' @param p        Total number of variables. Must be divisible by M.
#' @param M        Number of groups. Default 10.
#' @param s        Number of active variables per group. Default 1.
#' @param rho_in   Within-group equicorrelation in [0, 1). Default 0.7.
#' @param rho_out  Between-group correlation in [0, rho_in). Default 0.2.
#' @param snr      Target population SNR. Default 5.
#' @param sigma_eps  Noise standard deviation. Default 1.
#' @param seed     RNG seed. Default NULL.
#'
#' @return Standard GVS DGP list (see .dgp_output_gvs()), with
#'   params$rho_out added.
#'
#' @examples
#' dgp <- dgp_between_corr(n = 100, p = 200, M = 10, s = 1,
#'                          rho_in = 0.7, rho_out = 0.3, seed = 1)
#' # With corr_max = 0.5, groups should still be recovered (rho_out < corr_max)
#' # With corr_max = 0.25, groups will be merged (rho_out > corr_max)
dgp_between_corr <- function(n, p, M = 10, s = 1, rho_in = 0.7, rho_out = 0.2,
                              snr = 5, sigma_eps = 1, seed = NULL) {

  if (p %% M != 0)
    stop(paste0("'p' must be divisible by 'M'. Got p = ", p, ", M = ", M, "."))
  if (length(rho_out) != 1 || rho_out < 0 || rho_out >= rho_in)
    stop(paste0("'rho_out' must be in [0, rho_in) = [0, ", rho_in, ")."))

  b <- p %/% M
  .check_dgp_gvs(n, p, M, s, rho_in, snr, sigma_eps, b_min = b)
  if (!is.null(seed)) set.seed(seed)

  groups_list <- lapply(seq(M), function(m) ((m - 1L) * b + 1L):(m * b))
  Sigma       <- .block_equi_cov(p, groups_list, rho_in, rho_out = rho_out)
  X           <- .draw_X(n, Sigma)
  act         <- .make_active(p, groups_list, s, Sigma, snr, sigma_eps)
  eps         <- stats::rnorm(n)
  y           <- X %*% act$beta + sigma_eps * eps

  .dgp_output_gvs(X, y, act$beta, groups_list, act$active_set, Sigma,
                   sigma_eps,
                   params = list(n = n, p = p, M = M, s = s,
                                 rho_in = rho_in, rho_out = rho_out,
                                 snr = snr, sigma_eps = sigma_eps,
                                 seed = seed))
}
