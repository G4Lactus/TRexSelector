# ==============================================================================
# trex_da.R
#
# Dependency-Aware T-Rex (DA-TREX) Selector
#
# Public interface:
#   trex_da()               -- main entry point
#
# Internal helpers (not exported):
#   .check_trex_da()        -- input validation
#   .setup_da()             -- neighbourhood / correlation dispatcher
#   .da_correct()           -- DA deflation kernel
#   .compute_fdp()          -- Phi_prime + FDP_hat computation
#   select_var_fun_DA_BT()  -- variable selection for BT-style paths
#
# External dependencies (must be on the search path):
#   random_experiments(), Phi_prime_fun(), fdp_hat(),
#   add_dummies(), select_var_fun()
#
# Optional speed-up packages:
#   WGCNA        -- fast correlation matrix
#   fastcluster  -- fast hierarchical clustering (BT only)
# ==============================================================================


# ==============================================================================
# .check_trex_da
# ==============================================================================

.check_trex_da <- function(X, y, tFDR, K, max_num_dummies,
                            groups, rho_grid_labels,
                            parallel_process, parallel_max_cores,
                            eps) {

  if (!is.matrix(X))        stop("'X' must be a matrix.")
  if (!is.numeric(X))       stop("'X' only allows numerical values.")
  if (anyNA(X))             stop("'X' contains NAs.")
  if (!is.vector(drop(y)))  stop("'y' must be a vector.")
  if (!is.numeric(y))       stop("'y' only allows numerical values.")
  if (anyNA(y))             stop("'y' contains NAs.")
  if (nrow(X) != length(drop(y)))
    stop("Number of rows in X does not match length of y.")

  if (length(tFDR) != 1 || tFDR < 0 || tFDR > 1)
    stop("'tFDR' must be a number in [0, 1].")
  if (length(K) != 1 || K < 2 || K %% 1 != 0)
    stop("'K' must be an integer >= 2.")
  if (length(max_num_dummies) != 1 ||
      max_num_dummies < 1 ||
      max_num_dummies %% 1 != 0)
    stop("'max_num_dummies' must be an integer >= 1.")

  if (!is.null(groups)) {
    if (!is.list(groups))
      stop("'groups' must be a list of integer vectors, one per hierarchy level.")
    if (length(groups) < 1)
      stop("'groups' must contain at least one level.")
    if (!all(lengths(groups) == ncol(X)))
      stop("Each element of 'groups' must be an integer vector of length p = ncol(X).")
    if (!is.null(rho_grid_labels)) {
      if (!is.numeric(rho_grid_labels))
        stop("'rho_grid_labels' must be a numeric vector.")
      if (length(rho_grid_labels) != length(groups))
        stop("'rho_grid_labels' must have the same length as 'groups'.")
    }
  }

  if (parallel_process) {
    if (length(parallel_max_cores) != 1 ||
        parallel_max_cores %% 1 != 0 ||
        parallel_max_cores < 2)
      stop("'parallel_max_cores' must be an integer >= 2 for parallel processing.")
  }

  invisible(NULL)
}


# ==============================================================================
# .setup_da
# ==============================================================================

#' Dispatch to the appropriate neighbourhood / correlation setup route and
#' return a unified environment consumed by both calibration loops.
#'
#' @return A named list:
#'   use_BT_style  -- logical; TRUE for all 3D-array (BT-style) paths
#'   gr_j_list     -- nested list [j][rho_level] of group-mate indices
#'                    (NULL for AR1 / equi)
#'   rho_grid      -- numeric grid of grouping levels
#'                    (NULL for AR1 / equi)
#'   rho_grid_len  -- integer length of rho_grid (NULL for AR1 / equi)
#'   opt_point_BT  -- reference index into rho_grid (NULL for AR1 / equi)
#'   cor_coef      -- estimated or user-supplied correlation coefficient
#'   kap           -- AR(1) window half-width (NULL unless AR1)
.setup_da <- function(method, X, p,
                       groups, rho_grid_labels,
                       cor_coef, rho_thr_DA,
                       hc_dist, hc_grid_length) {

  use_BT_style <- !is.null(groups) || method %in% c("trex+DA+BT", "trex+DA+NN")
  kap          <- NULL

  # ── Route 1: prior-groups ──────────────────────────────────────────────────
  if (!is.null(groups)) {
    rho_grid_len <- length(groups)
    rho_grid     <- if (!is.null(rho_grid_labels)) rho_grid_labels
                    else                           seq_along(groups)
    opt_point_BT <- round(0.75 * rho_grid_len)

    gr_j_list <- lapply(seq(1, p), function(j) {
      lapply(seq(1, rho_grid_len), function(x) {
        which(groups[[x]] == groups[[x]][j] & seq(1, p) != j)
      })
    })

    return(list(use_BT_style = use_BT_style,
                gr_j_list    = gr_j_list,
                rho_grid     = rho_grid,
                rho_grid_len = rho_grid_len,
                opt_point_BT = opt_point_BT,
                cor_coef     = cor_coef,
                kap          = kap))
  }

  # ── Route 2: BT dendrogram ────────────────────────────────────────────────
  if (method == "trex+DA+BT") {
    cor_mat <- if (system.file(package = "WGCNA") != "") {
      WGCNA::cor(X)
    } else {
      message("Install 'WGCNA' to speed up correlation computation.")
      stats::cor(X)
    }

    dendrogram <- if (system.file(package = "fastcluster") != "") {
      fastcluster::hclust(stats::as.dist(1 - abs(cor_mat)), method = hc_dist)
    } else {
      message("Install 'fastcluster' to speed up hierarchical clustering.")
      stats::hclust(stats::as.dist(1 - abs(cor_mat)), method = hc_dist)
    }

    rho_grid_subsample <- round(seq(1, p, length.out = hc_grid_length))
    rho_grid_len       <- hc_grid_length
    rho_grid           <- c(1 - rev(dendrogram$height), 1)[rho_grid_subsample]
    opt_point_BT       <- round(0.75 * rho_grid_len)
    clusters           <- stats::cutree(dendrogram, h = 1 - rho_grid)

    gr_j_list <- lapply(seq(1, p), function(j) {
      lapply(seq(1, rho_grid_len), function(x) {
        gr_j <- which(clusters[, x] == clusters[j, x])
        gr_j[-which(gr_j == j)]
      })
    })

    return(list(use_BT_style = use_BT_style,
                gr_j_list    = gr_j_list,
                rho_grid     = rho_grid,
                rho_grid_len = rho_grid_len,
                opt_point_BT = opt_point_BT,
                cor_coef     = cor_coef,
                kap          = kap))
  }

  # ── Route 3: NN threshold sweep ───────────────────────────────────────────
  if (method == "trex+DA+NN") {
    cor_mat <- if (system.file(package = "WGCNA") != "") {
      WGCNA::cor(X)
    } else {
      message("Install 'WGCNA' to speed up correlation computation.")
      stats::cor(X)
    }

    rho_grid_len <- hc_grid_length
    rho_grid     <- seq(0, 1, length.out = rho_grid_len)
    opt_point_BT <- round(0.75 * rho_grid_len)

    gr_j_list <- lapply(seq(1, p), function(j) {
      lapply(seq(1, rho_grid_len), function(x) {
        gr_j <- which(abs(cor_mat[, j]) >= rho_grid[x])
        gr_j[-which(gr_j == j)]
      })
    })

    return(list(use_BT_style = use_BT_style,
                gr_j_list    = gr_j_list,
                rho_grid     = rho_grid,
                rho_grid_len = rho_grid_len,
                opt_point_BT = opt_point_BT,
                cor_coef     = cor_coef,
                kap          = kap))
  }

  # ── Route 4: AR1 ─────────────────────────────────────────────────────────
  if (method == "trex+DA+AR1") {
    if (is.na(cor_coef)) {
      cor_coef <- abs(mean(apply(X, 1, function(smpl) {
        stats::coef(stats::arima(smpl, order = c(1, 0, 0),
                                 include.mean = FALSE, method = "ML"))
      })))
    }
    kap <- ceiling(log(rho_thr_DA) / log(cor_coef))

    return(list(use_BT_style = FALSE,
                gr_j_list    = NULL,
                rho_grid     = NULL,
                rho_grid_len = NULL,
                opt_point_BT = NULL,
                cor_coef     = cor_coef,
                kap          = kap))
  }

  # ── Route 5: equi ─────────────────────────────────────────────────────────
  if (method == "trex+DA+equi") {
    if (is.na(cor_coef)) {
      cor_mat  <- if (system.file(package = "WGCNA") != "") {
        WGCNA::cor(X)
      } else {
        stats::cor(X)
      }
      cor_coef <- mean(cor_mat[lower.tri(cor_mat, diag = FALSE)])
    }

    return(list(use_BT_style = FALSE,
                gr_j_list    = NULL,
                rho_grid     = NULL,
                rho_grid_len = NULL,
                opt_point_BT = NULL,
                cor_coef     = cor_coef,
                kap          = NULL))
  }
}


# ==============================================================================
# .da_correct
# ==============================================================================

#' Apply the dependency-aware deflation for all routes.
#'
#' @return List: phi_T_mat, Phi, phi_T_array_BT (NULL for AR1/equi),
#'               Phi_BT (NULL for AR1/equi).
.da_correct <- function(method, phi_T_mat, Phi, T_stop, p,
                        cor_coef, rho_thr_DA, kap,
                        gr_j_list, rho_grid_len, use_BT_style) {

  phi_T_array_BT <- NULL
  Phi_BT         <- NULL

  # AR1 -----------------------------------------------------------------------
  if (method == "trex+DA+AR1") {

    DA_delta_mat <- matrix(NA, nrow = p, ncol = T_stop)
    for (t in seq(T_stop)) {
      for (j in seq(1, p)) {
        sw <- c(seq(max(1, j - kap), max(1, j - 1)),
                seq(min(p, j + 1),   min(p, j + kap)))
        if (j %in% c(1, p)) sw <- sw[-which(sw == j)]
        DA_delta_mat[j, t] <- 2 - min(abs(phi_T_mat[j, t] - phi_T_mat[sw, t]))
      }
    }

    phi_T_mat <- phi_T_mat / DA_delta_mat
    Phi       <- Phi / DA_delta_mat[, T_stop]
  }

  # equi ----------------------------------------------------------------------
  if (method == "trex+DA+equi" && abs(cor_coef) > rho_thr_DA) {

    DA_delta_mat <- matrix(NA, nrow = p, ncol = T_stop)
    for (t in seq(T_stop)) {
      for (j in seq(1, p)) {
        sw                 <- seq(1, p)[-j]
        DA_delta_mat[j, t] <- 2 - min(abs(phi_T_mat[j, t] - phi_T_mat[sw, t]))
      }
    }

    phi_T_mat <- phi_T_mat / DA_delta_mat
    Phi       <- Phi       / DA_delta_mat[, T_stop]
  }

  # BT / NN / prior-groups ----------------------------------------------------
  if (use_BT_style) {

    DA_delta_mat_BT <- matrix(NA, nrow = p, ncol = rho_grid_len)
    for (j in seq(1, p)) {
      DA_delta_mat_BT[j, ] <- sapply(gr_j_list[[j]], function(x) {
        if (length(x) == 0) 2
        else 2 - min(abs(phi_T_mat[j, T_stop] - phi_T_mat[x, T_stop]))
      })
    }

    phi_T_array_BT <- array(apply(DA_delta_mat_BT, 2, function(x) phi_T_mat / x),
                            dim = c(p, T_stop, rho_grid_len))

    Phi_BT <- Phi / DA_delta_mat_BT
  }

  list(phi_T_mat      = phi_T_mat,
       Phi            = Phi,
       phi_T_array_BT = phi_T_array_BT,
       Phi_BT         = Phi_BT)
}


# ==============================================================================
# .compute_fdp
# ==============================================================================

#' Compute Phi_prime and FDP_hat for BT-style (matrix) and AR1/equi (vector).
#'
#' @return List: Phi_prime, FDP_hat.
.compute_fdp <- function(use_BT_style, p, T_stop, num_dummies,
                          phi_T_mat, phi_T_array_BT, Phi, Phi_BT,
                          V, V_len, rho_grid_len, eps) {
  if (use_BT_style) {
    Phi_prime <- matrix(
      sapply(seq(rho_grid_len), function(x) {
        Phi_prime_fun(p = p, T_stop = T_stop, num_dummies = num_dummies,
                      phi_T_mat = matrix(phi_T_array_BT[, , x], nrow = p, ncol = T_stop),
                      Phi = Phi_BT[, x], eps = eps)
      }), nrow = p, ncol = rho_grid_len)

    FDP_hat <- matrix(
      sapply(seq(rho_grid_len), function(x) {
        fdp_hat(V = V, Phi = Phi_BT[, x], Phi_prime = Phi_prime[, x])
      }), nrow = V_len, ncol = rho_grid_len)

  } else {
    Phi_prime <- Phi_prime_fun(p = p, T_stop = T_stop, num_dummies = num_dummies,
                                phi_T_mat = phi_T_mat, Phi = Phi, eps = eps)
    FDP_hat   <- fdp_hat(V = V, Phi = Phi, Phi_prime = Phi_prime)
  }

  list(Phi_prime = Phi_prime, FDP_hat = FDP_hat)
}


# ==============================================================================
# select_var_fun_DA_BT
# ==============================================================================

#' Variable selection for BT-style paths (BT, NN, prior-groups).
#'
#' Finds the joint optimum over (T, V, rho_grid) that maximises the number
#' of selected variables while keeping the FDP estimate at or below tFDR.
#'
#' @param p                 Number of candidate variables.
#' @param tFDR              Target FDR level.
#' @param T_stop            Number of included dummies at stopping.
#' @param FDP_hat_array_BT  FDP estimate array, dim = (T_stop x V_len x rho_grid_len).
#' @param Phi_array_BT      Deflated relative occurrences, same layout.
#' @param V                 Voting level grid.
#' @param rho_grid          Grouping-level grid.
#'
#' @return List: selected_var, v_thresh, rho_thresh, R_array.
select_var_fun_DA_BT <- function(p, tFDR, T_stop,
                                 FDP_hat_array_BT, Phi_array_BT,
                                 V, rho_grid) {
  if (T_stop > 1) {
    FDP_hat_array_BT <- array(FDP_hat_array_BT[-T_stop, , ],
                              dim = dim(FDP_hat_array_BT) - c(1, 0, 0))
    Phi_array_BT     <- array(Phi_array_BT[-T_stop, , ],
                              dim = dim(Phi_array_BT)     - c(1, 0, 0))
  }

  R_array <- array(NA, dim = dim(FDP_hat_array_BT))
  for (TT in seq(dim(FDP_hat_array_BT)[1]))
    for (VV in seq_along(V))
      for (RR in seq(dim(FDP_hat_array_BT)[3]))
        R_array[TT, VV, RR] <- sum(Phi_array_BT[TT, , RR] > V[VV])

  FDP_hat_array_BT[FDP_hat_array_BT > tFDR] <- Inf
  val_max    <- suppressWarnings(max(R_array[!is.infinite(FDP_hat_array_BT)]))
  ind_max    <- matrix(which(R_array == val_max, arr.ind = TRUE), ncol = 3)
  ind_thresh <- matrix(ind_max[which(ind_max[, 2] == max(ind_max[, 2])), ], ncol = 3)
  ind_thresh <- ind_thresh[nrow(ind_thresh), ]

  selected_var    <- rep(0L, times = p)
  selected_var[which(Phi_array_BT[ind_thresh[1], , ind_thresh[3]] > V[ind_thresh[2]])] <- 1L

  list(selected_var = selected_var,
       v_thresh     = V[ind_thresh[2]],
       rho_thresh   = rho_grid[ind_thresh[3]],
       R_array      = R_array)
}


# ==============================================================================
# trex_da  (main entry point)
# ==============================================================================

#' Run the Dependency-Aware T-Rex (DA-TREX) Selector
#'
#' Performs fast FDR-controlled variable selection in high-dimensional
#' settings while accounting for predictor correlation structure.
#' The dependency structure may be supplied as prior group knowledge
#' (recommended when known) or estimated automatically from X.
#'
#' @param X               Real-valued predictor matrix (n x p).
#' @param y               Response vector (length n).
#' @param tFDR            Target FDR level in [0, 1]. Default 0.2.
#' @param K               Number of random experiments (integer >= 2).
#'                        Default 20.
#' @param max_num_dummies Integer multiplier; max dummies = factor * p.
#'                        Default 10.
#' @param max_T_stop      If TRUE, max T_stop = ceiling(n / 2). Default TRUE.
#' @param method          Dependency model when groups = NULL. One of:
#'                        'trex+DA+BT'   (default), 'trex+DA+NN',
#'                        'trex+DA+AR1', 'trex+DA+equi'.
#' @param type            'lar' (default) or 'lasso'.
#'
#' @param groups          Prior group knowledge: a list of L integer vectors
#'                        of length p, ordered \strong{coarse to fine}
#'                        (groups[[1]] = fewest/largest clusters,
#'                        groups[[L]] = most/smallest clusters).
#'                        This matches the internal rho_grid orientation
#'                        of the BT and NN routes (x = 1 is coarsest,
#'                        x = L is finest). When supplied, method, hc_dist,
#'                        and hc_grid_length are ignored. Full 3D calibration
#'                        is preserved across all L levels.
#'                        Default NULL (auto-compute from X).
#' @param rho_grid_labels Optional numeric vector of length L: semantic
#'                        labels for each level, in the same coarse-to-fine
#'                        order as groups (e.g. increasing correlation
#'                        thresholds: c(0.2, 0.5, 0.8)). Defaults to
#'                        seq_along(groups). Only used when groups != NULL.
#'
#' @param cor_coef        AR(1) or equicorrelation coefficient.
#'                        Estimated from X if NA (default).
#' @param rho_thr_DA      Equi correction skipped when |cor_coef| <= this.
#'                        Default 0.02.
#' @param hc_dist         Hierarchical clustering linkage for BT:
#'                        'single' (default), 'complete', 'average',
#'                        'mcquitty', 'ward.D', 'ward.D2'.
#' @param hc_grid_length  Number of dendrogram / NN grid points. Default
#'                        min(20, p).
#'
#' @param parallel_process    Logical. Default FALSE.
#' @param parallel_max_cores  Max cores for parallel execution.
#' @param seed            RNG seed (ignored when parallel_process = FALSE).
#' @param eps             Numerical zero. Default .Machine$double.eps.
#' @param verbose         Print progress. Default TRUE.
#'
#' @return Named list containing selected_var, calibration outputs
#'         (T_stop, num_dummies, V, rho_grid), FDP/Phi arrays or matrices,
#'         Phi_prime, and all parameters used.
#'
#' @importFrom parallel detectCores makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach getDoParWorkers registerDoSEQ
#' @importFrom stats coef arima cor as.dist hclust cutree
#' @export
#'
#' @examples
#' set.seed(1234)
#' n <- 50; p <- 100
#' X    <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
#' beta <- c(rep(3, 3), rep(0, 97))
#' y    <- X %*% beta + stats::rnorm(n)
#'
#' # Auto-computed BT dendrogram
#' res <- trex_da(X = X, y = y, method = "trex+DA+BT", K = 20)
#'
#' # Prior group knowledge (3 levels, p = 6, coarse to fine)
#' # Level 1: 2 groups of 3  (coarsest, low rho threshold)
#' # Level 2: 3 groups of 2  (medium)
#' # Level 3: 6 singletons   (finest, high rho threshold)
#' grps <- list(c(1,1,1,2,2,2), c(1,1,2,3,4,4), c(1,2,3,4,5,6))
#' res2 <- trex_da(X = X[, 1:6], y = y, groups = grps,
#'                 rho_grid_labels = c(0.2, 0.5, 0.9), K = 20)
trex_da <- function(X,
                    y,
                    tFDR               = 0.2,
                    K                  = 20,
                    max_num_dummies    = 10,
                    max_T_stop         = TRUE,
                    method             = "trex+DA+BT",
                    type               = "lar",
                    groups             = NULL,
                    rho_grid_labels    = NULL,
                    cor_coef           = NA,
                    rho_thr_DA         = 0.02,
                    hc_dist            = "single",
                    hc_grid_length     = min(20, ncol(X)),
                    parallel_process   = FALSE,
                    parallel_max_cores = min(K, max(1L, parallel::detectCores(logical = FALSE))),
                    seed               = NULL,
                    eps                = .Machine$double.eps,
                    verbose            = TRUE) {

  # ── Validate and coerce inputs ──────────────────────────────────────────────

  method  <- match.arg(method,
                       c("trex+DA+AR1", "trex+DA+equi", "trex+DA+BT", "trex+DA+NN"))
  type    <- match.arg(type, c("lar", "lasso"))

  if (method == "trex+DA+BT" && is.null(groups)) {
    hc_dist <- match.arg(hc_dist,
                         c("single", "complete", "average",
                           "mcquitty", "ward.D", "ward.D2"))
    if (hc_dist %in% c("ward.D", "ward.D2"))
      warning("Ward linkage is designed for Euclidean distances. ",
              "With d = 1 - |rho| the within-cluster variance ",
              "interpretation does not hold. Proceeding anyway.")
  }

  .check_trex_da(X, y, tFDR, K, max_num_dummies,
                 groups, rho_grid_labels,
                 parallel_process, parallel_max_cores, eps)

  # Silently clamp parallel_max_cores
  if (parallel_process) {
    hard_limit <- min(K, max(1L, parallel::detectCores(logical = FALSE)))
    if (parallel_max_cores > hard_limit) {
      message("Setting parallel_max_cores = ", hard_limit, " (physical cores).\n")
      parallel_max_cores <- hard_limit
    }
  }

  # ── Dimensions and preprocessing ────────────────────────────────────────────

  n <- nrow(X);  p <- ncol(X)
  if (p > 1e4) warning("Feature space > 10,000 variables. RAM issues may occur.")

  X <- scale(X);  y <- y - mean(y)

  # ── Build neighbourhood / correlation structure ──────────────────────────────

  da <- .setup_da(method, X, p, groups, rho_grid_labels,
                  cor_coef, rho_thr_DA, hc_dist, hc_grid_length)

  # Unpack for readability inside the loops
  use_BT_style <- da$use_BT_style
  rho_grid_len <- da$rho_grid_len
  rho_grid     <- da$rho_grid
  opt_point_BT <- da$opt_point_BT
  cor_coef     <- da$cor_coef
  kap          <- da$kap
  gr_j_list    <- da$gr_j_list

  # ── Voting grid and stopping-rule reference point ───────────────────────────

  V         <- seq(0.5, 1 - eps, by = 1 / K)
  V_len     <- length(V)
  opt_point <- which(abs(V - 0.75) < eps)
  if (length(opt_point) == 0) opt_point <- length(V[V < 0.75])

  fdp_ref <- function(FDP_hat)       # FDP at the 75% calibration reference
    if (use_BT_style) FDP_hat[opt_point, opt_point_BT] else FDP_hat[opt_point]

  fdp_edge <- function(FDP_hat)      # FDP at the most conservative voting level
    if (use_BT_style) FDP_hat[V_len, opt_point_BT]    else FDP_hat[V_len]

  # ── Parallel cluster ────────────────────────────────────────────────────────

  if (parallel_process && foreach::getDoParWorkers() == 1) {
    cl <- parallel::makeCluster(parallel_max_cores)
    doParallel::registerDoParallel(cl)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    on.exit(foreach::registerDoSEQ(),  add = TRUE)
  }

  # ── Helper: run K random experiments and apply full DA pipeline ─────────────

  .run_step <- function(T_stop, num_dummies, lars_state_list = NULL) {
    suppressWarnings(
      rand_exp <- random_experiments(
        X = X, y = y, K = K, T_stop = T_stop, num_dummies = num_dummies,
        method = method, type = type, early_stop = TRUE,
        lars_state_list = lars_state_list, verbose = verbose,
        intercept = FALSE, standardize = TRUE,
        parallel_process = parallel_process,
        parallel_max_cores = parallel_max_cores,
        seed = seed, eps = eps
      )
    )
    da_res  <- .da_correct(method, rand_exp$phi_T_mat, rand_exp$Phi,
                            T_stop, p, cor_coef, rho_thr_DA, kap,
                            gr_j_list, rho_grid_len, use_BT_style)
    fdp_res <- .compute_fdp(use_BT_style, p, T_stop, num_dummies,
                             da_res$phi_T_mat, da_res$phi_T_array_BT,
                             da_res$Phi, da_res$Phi_BT,
                             V, V_len, rho_grid_len, eps)
    list(rand_exp       = rand_exp,
         phi_T_mat      = da_res$phi_T_mat,
         Phi            = da_res$Phi,
         phi_T_array_BT = da_res$phi_T_array_BT,
         Phi_BT         = da_res$Phi_BT,
         Phi_prime      = fdp_res$Phi_prime,
         FDP_hat        = fdp_res$FDP_hat)
  }

  # ── L-loop: find sufficient num_dummies ─────────────────────────────────────

  LL      <- 1L;  T_stop <- 1L
  FDP_hat <- if (use_BT_style) matrix(NA, V_len, rho_grid_len) else rep(NA, V_len)

  while ((LL <= max_num_dummies && fdp_ref(FDP_hat) > tFDR) ||
         sum(!is.na(FDP_hat)) == 0) {
    num_dummies <- LL * p;  LL <- LL + 1L
    step        <- .run_step(T_stop, num_dummies)
    FDP_hat     <- step$FDP_hat
    if (verbose) cat("\n  Appended dummies:", num_dummies, "\n")
  }

  # ── T-loop: extend T_stop while FDP is below target ─────────────────────────

  acc <- if (use_BT_style) {
    list(FDP = array(FDP_hat,    dim = c(dim(FDP_hat),    1)),
         Phi = array(step$Phi_BT, dim = c(dim(step$Phi_BT), 1)))
  } else {
    list(FDP = matrix(FDP_hat, nrow = 1),
         Phi = matrix(step$Phi, nrow = 1))
  }

  max_T <- if (max_T_stop) min(num_dummies, ceiling(n / 2)) else num_dummies
  if (!is.null(seed)) seed <- seed + 12345L

  while (fdp_edge(FDP_hat) <= tFDR && T_stop < max_T) {
    T_stop <- T_stop + 1L
    step   <- .run_step(T_stop, num_dummies, step$rand_exp$lars_state_list)
    FDP_hat <- step$FDP_hat

    acc <- if (use_BT_style) {
      list(FDP = array(c(acc$FDP, FDP_hat),    dim = dim(acc$FDP) + c(0, 0, 1)),
           Phi = array(c(acc$Phi, step$Phi_BT), dim = dim(acc$Phi) + c(0, 0, 1)))
    } else {
      list(FDP = rbind(acc$FDP, FDP_hat),
           Phi = rbind(acc$Phi, step$Phi))
    }
    if (verbose) cat("\n  Included dummies before stopping:", T_stop, "\n")
  }

  # ── Variable selection ────────────────────────────────────────────────────

  if (use_BT_style) {
    FDP_arr  <- aperm(acc$FDP, c(3, 1, 2))
    Phi_arr  <- aperm(acc$Phi, c(3, 1, 2))
    sel      <- select_var_fun_DA_BT(p, tFDR, T_stop, FDP_arr, Phi_arr, V, rho_grid)
    rho_thresh <- sel$rho_thresh;  R_out <- sel$R_array
  } else {
    sel        <- select_var_fun(p, tFDR, T_stop, acc$FDP, acc$Phi, V)
    rho_thresh <- NA;              R_out <- sel$R_mat
  }

  # ── Assemble and return ───────────────────────────────────────────────────

  base <- list(selected_var = sel$selected_var,
               tFDR         = tFDR,
               T_stop       = T_stop,
               num_dummies  = num_dummies,
               V            = V,
               v_thresh     = sel$v_thresh,
               rho_thresh   = rho_thresh,
               Phi_prime    = step$Phi_prime,
               method       = method,
               type         = type,
               cor_coef     = cor_coef,
               rho_thr_DA   = rho_thr_DA,
               hc_dist      = hc_dist,
               groups       = groups)

  extra <- if (use_BT_style)
    list(rho_grid         = rho_grid,
         FDP_hat_array_BT = FDP_arr,
         Phi_array_BT     = Phi_arr,
         R_array          = R_out,
         phi_T_array_BT   = step$phi_T_array_BT)
  else
    list(FDP_hat_mat = acc$FDP,
         Phi_mat     = acc$Phi,
         R_mat       = R_out,
         phi_T_mat   = step$phi_T_mat)

  return(c(base, extra))
}
