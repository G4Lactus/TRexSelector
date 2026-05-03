# ==============================================================================
# trex_GVS.R
#
# T-Rex+GVS selector — Grouped Variable Selection
# Reference: Machkour et al. (2022), doi:10.23919/EUSIPCO55093.2022.9909883
#
# Public interface:
#   trex_GVS()           — main entry point
#
# Internal helpers (.):
#   .check_trex_gvs()    — input validation
#   .setup_gvs()         — clustering / group setup         [called ONCE]
#   .compute_lambda2()   — lambda_2 via ridge CV            [called ONCE]
#   .draw_dummies_gvs()  — random dummy draws per experiment [only stochastic part]
#   .augment_en()        — EN  augmentation policy
#   .augment_ien()       — IEN augmentation policy          [add future types here]
#   .lm_dummy_gvs()      — one random experiment (dispatches augmentation)
#   .random_exp_gvs()    — K random experiments (serial or parallel)
#
# External dependencies (must be on the search path):
#   Phi_prime_fun(), fdp_hat(), select_var_fun()
#
# Notation (consistent with Machkour et al., CAMSAP 2023):
#   X          — original predictor matrix, n x p
#   X_tilde    — extended matrix [X | dummy_block], n x p_d, p_d = p + L
#   L          — number of dummy columns  (= num_dummies)
#   M          — number of variable groups (clusters)
#   p_m        — size of group m,  m = 1,...,M
#   1_m        — binary support vector of group m, length p
#   lambda_2   — ridge / IEN penalty parameter (= lambda_2_lars)
#
# Augmentation policy contract (uniform interface for all current and future
# Tikhonov-type augmentation strategies):
#   Input : X_tilde      (n x p_d real matrix, p_d = p + L)
#           y            (response vector, length n)
#           lambda_2_lars (positive scalar)
#           gvs_setup    (output of .setup_gvs(); ignored by policies
#                         that do not require group structure)
#   Output: list(X_aug, y_aug)
# ==============================================================================


# ==============================================================================
# .check_trex_gvs
# ==============================================================================

.check_trex_gvs <- function(X, y, tFDR, K, max_num_dummies,
                            groups, group_labels,
                            corr_max, hc_method, lambda_2_lars,
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

  p <- ncol(X)

  if (!is.null(groups)) {
    if (!is.numeric(groups) && !is.integer(groups))
      stop("'groups' must be a numeric or integer vector.")
    if (length(groups) != p)
      stop("'groups' must have length p = ncol(X).")
    if (any(groups != floor(groups)) || any(groups < 1L))
      stop("'groups' must contain positive integers (cluster indices).")
    if (!is.null(group_labels)) {
      if (!is.character(group_labels))
        stop("'group_labels' must be a character vector.")
      if (length(group_labels) != length(unique(groups)))
        stop("'group_labels' must have one entry per unique value in 'groups'.")
    }
  } else {
    if (length(corr_max) != 1 || corr_max < 0 || corr_max > 1)
      stop("'corr_max' must be a number in [0, 1].")
    valid_hc_methods <- c("single", "complete", "average", "mcquitty")
    if (length(hc_method) != 1 || !hc_method %in% valid_hc_methods)
      stop("'hc_method' must be one of: ",
           paste0("'", valid_hc_methods, "'", collapse = ", "), ".\n",
           "  'ward.D2' is intentionally excluded: Ward's criterion requires ",
           "Euclidean distances and is incompatible with the d = 1 - |cor| ",
           "metric used here. See ?trex_GVS for details.")
  }

  if (!is.null(lambda_2_lars)) {
    if (length(lambda_2_lars) != 1 || lambda_2_lars < eps)
      stop("'lambda_2_lars' must be a positive number.")
  }

  if (parallel_process) {
    if (length(parallel_max_cores) != 1 ||
          parallel_max_cores %% 1 != 0 ||
          parallel_max_cores < 2)
      stop("'parallel_max_cores' must be an integer >= 2.")
  }

  invisible(NULL)
}


# ==============================================================================
# .setup_gvs
# ==============================================================================

#' Compute the full group structure of X.
#'
#' Called ONCE in trex_GVS(). All quantities returned are deterministic
#' functions of X (or the user-supplied groups) — none of them should ever
#' be recomputed inside the random experiments.
#'
#' @return Named list:
#'   clusters_list     — list of integer index vectors, one per cluster
#'   cluster_sizes     — integer vector of per-cluster sizes  (p_1,...,p_M)
#'   sigma_sub_list    — list of per-cluster sample covariance matrices
#'   IEN_cl_id_vectors — (M x p) binary matrix; row m = support vector 1_m
#'   max_clusters      — M, total number of clusters
#'   group_labels      — character vector of cluster names, or NULL
#'   hc_method         — linkage method used (recorded for provenance)
.setup_gvs <- function(X, corr_max, hc_method, groups, group_labels) {

  p <- ncol(X)

  if (!is.null(groups)) {
    # ── Route 1: prior group knowledge ─────────────────────────────────────────
    # groups[j] = cluster index of variable j  (positive integer)
    clusters_list <- split(seq(p), groups)
    max_clusters  <- length(clusters_list)
    cluster_sizes <- lengths(clusters_list)

  } else {
    # ── Route 2: single-linkage hierarchical clustering of X ───────────────────
    sigma_X      <- stats::cov(X)
    # Distance metric: d(j, j') = 1 - |cor(x_j, x_j')| in [0, 1].
    # This is NOT a Euclidean distance. Consequently, 'ward.D2' is excluded
    # from hc_method: Ward's criterion requires Euclidean geometry and
    # produces meaningless clusters with a correlation-based distance matrix.
    # See @param hc_method in trex_GVS() for details. If Ward clustering is
    # needed, compute it externally and pass the result via 'groups'.
    sigma_X_dist <- stats::as.dist(1 - abs(stats::cov2cor(sigma_X)))
    fit          <- stats::hclust(sigma_X_dist, method = hc_method)
    clusters_vec <- stats::cutree(fit, h = 1 - corr_max)
    max_clusters <- max(clusters_vec)

    clusters_list <- split(seq(p), clusters_vec)
    cluster_sizes <- lengths(clusters_list)
  }

  # ── Per-cluster sample covariance matrices ────────────────────────────────────
  # Computed here ONCE from the (already scaled) X.
  # Passed as gvs_setup$sigma_sub_list into .draw_dummies_gvs() — never
  # recomputed inside the random experiments.
  sigma_sub_list <- lapply(clusters_list, function(idx) {
    stats::cov(X[, idx, drop = FALSE])
  })

  # ── Binary support matrix for IEN — (M x p) ──────────────────────────────────
  # Row m is the support vector 1_m in {0,1}^p:
  #   1_m[j] = 1 iff variable j belongs to group m.
  IEN_cl_id_vectors <- t(vapply(seq(max_clusters), function(z) {
    v                 <- rep(FALSE, times = p)
    v[clusters_list[[z]]] <- TRUE
    v
  }, logical(p)))

  list(
    clusters_list     = clusters_list,
    cluster_sizes     = cluster_sizes,
    sigma_sub_list    = sigma_sub_list,
    IEN_cl_id_vectors = IEN_cl_id_vectors,
    max_clusters      = max_clusters,
    group_labels      = group_labels,
    hc_method         = if (!is.null(groups)) NA_character_ else hc_method
  )
}


# ==============================================================================
# .compute_lambda2
# ==============================================================================

#' Determine lambda_2 via 10-fold cross-validated ridge regression.
#'
#' Called ONCE in trex_GVS(). The resulting scalar is fixed and shared
#' across all K random experiments and all L / T iterations.
#'
#' Conversion from glmnet scale to lars scale:
#'   lambda_lars = lambda_glmnet * p / 2
#' where p = ncol(X) and alpha = 0 (pure ridge, 1 - alpha = 1).
.compute_lambda2 <- function(X, y, intercept, standardize) {
  p     <- ncol(X)
  cvfit <- glmnet::cv.glmnet(
    x            = X,
    y            = y,
    intercept    = intercept,
    standardize  = standardize,
    alpha        = 0,
    type.measure = "mse",
    family       = "gaussian",
    nfolds       = 10
  )
  cvfit$lambda.1se * p / 2
}


# ==============================================================================
# .draw_dummies_gvs
# ==============================================================================

#' Draw cluster-aware dummy predictors for one random experiment.
#'
#' Generates the extended matrix X_tilde = [X | dummy_block] in R^{n x p_d},
#' where p_d = p + L and L = num_dummies. The dummy block consists of
#' L / p = w layers; each layer draws one dummy vector per cluster from
#' N(0, Sigma_m), where Sigma_m = cov(X_{C_m}) is the pre-computed
#' within-cluster covariance (gvs_setup$sigma_sub_list[[m]]).
#'
#' This is the ONLY stochastic part of the GVS pipeline. All structural
#' quantities (clustering, Sigma_m, IEN vectors, lambda_2) are deterministic
#' and pre-computed in .setup_gvs() / .compute_lambda2().
#'
#' @param X          Scaled predictor matrix (n x p).
#' @param gvs_setup  Output of .setup_gvs().
#' @param num_dummies Number of dummies L (must be a multiple of p).
#'
#' @return X_tilde: matrix of size n x (p + L).
.draw_dummies_gvs <- function(X, gvs_setup, num_dummies) {

  n              <- nrow(X)
  p              <- ncol(X)
  w_max          <- num_dummies / p         # number of dummy layers
  max_clusters   <- gvs_setup$max_clusters
  cluster_sizes  <- gvs_setup$cluster_sizes
  sigma_sub_list <- gvs_setup$sigma_sub_list

  idx           <- cumsum(cluster_sizes)    # column-end index per cluster
  X_p_sub_dummy <- matrix(NA_real_, nrow = n, ncol = p)
  X_tilde       <- matrix(NA_real_, nrow = n, ncol = p + num_dummies)
  X_tilde[, seq(p)] <- X

  for (w in seq(w_max)) {
    for (z in seq(max_clusters)) {
      cols <- if (z == 1L) seq(idx[z]) else seq(idx[z - 1L] + 1L, idx[z])
      X_p_sub_dummy[, cols] <- MASS::mvrnorm(
        n         = n,
        mu        = rep(0, times = cluster_sizes[z]),
        Sigma     = sigma_sub_list[[z]],
        empirical = FALSE
      )
    }
    X_tilde[, seq(w * p + 1L, (w + 1L) * p)] <- X_p_sub_dummy
  }

  X_tilde
}


# ==============================================================================
# Augmentation policies
#
# Uniform interface — see file header for full contract.
#
# Add future Tikhonov-type policies here following the same signature.
# Register new types in the match.arg() call in trex_GVS() and the
# switch() dispatcher in .lm_dummy_gvs().
# ==============================================================================

#' Ordinary Elastic Net (EN) augmentation  [Zou & Hastie, JRSSB 2005].
#'
#' Reformulates the EN objective as a Lasso-type problem by appending a
#' scaled identity block below X_tilde. Applied to the full extended matrix
#' X_tilde = [X | dummy_block] in R^{n x p_d}, p_d = p + L.
#'
#' Augmented system (Lasso-type, eq. (1) in Machkour et al. CAMSAP 2023):
#'
#'   X'_EN = (1 / sqrt(1 + lambda_2)) *
#'             [ X_tilde            ]   <- n      rows
#'             [ sqrt(lambda_2)*I_pd ]  <- p_d    rows   (isotropic ridge)
#'
#'   y'_EN = [ y ; 0_{p_d} ]
#'
#' The bottom block has p_d rows — one per column of X_tilde. No group
#' structure is exploited; every coefficient is penalized equally.
#'
#' @param X_tilde     Extended matrix [X | dummy_block], n x p_d.
#' @param y           Response vector, length n.
#' @param lambda_2_lars Ridge penalty parameter lambda_2 > 0.
#' @param gvs_setup   Output of .setup_gvs() — unused by EN, present for
#'                    interface uniformity.
#' @return list(X_aug, y_aug).
.augment_en <- function(X_tilde, y, lambda_2_lars, gvs_setup) {
  p_d   <- ncol(X_tilde)
  X_aug <- (1 / sqrt(1 + lambda_2_lars)) *
    rbind(X_tilde,
          diag(rep(sqrt(lambda_2_lars), times = p_d)))
  y_aug <- append(y, rep(0, times = p_d))
  list(X_aug = X_aug, y_aug = y_aug)
}


#' Informed Elastic Net (IEN) augmentation  [Machkour et al., CAMSAP 2023].
#'
#' Reformulates the IEN objective (Definition 1 in the paper) as a
#' Lasso-type problem (Theorem 1) by appending M cluster-structured rows
#' below X_tilde. Applied to the full extended matrix
#' X_tilde = [X | dummy_block] in R^{n x p_d}, p_d = p + L.
#'
#' IEN Lagrangian:
#'
#'   L_IEN(beta) = ||y - X_tilde * beta||^2_2
#'               + lambda_1 * ||beta||_1
#'               + lambda_2 * sum_{m=1}^{M} (1_m' * beta)^2 / p_m
#'
#' where 1_m in {0,1}^p_d is the support vector of group m extended across
#' all dummy layers, and p_m = |G_m| is the group size.
#'
#' Augmented system (Theorem 1 extended to X_tilde):
#'
#'   X'_IEN = sqrt(lambda_2) *
#'              [ X_tilde / sqrt(lambda_2)          ]  <- n rows
#'              [ (1/sqrt(p_1)) * [1_1'|1_1'|...|1_1'] ]  \
#'              [         ...                        ]   > M rows
#'              [ (1/sqrt(p_M)) * [1_M'|1_M'|...|1_M'] ]  /
#'
#'   y'_IEN = [ y ; 0_M ]
#'
#' The bottom block has M rows — one per group. Each row replicates the
#' binary support vector 1_m across all (w+1) column blocks of X_tilde
#' (original block + w dummy layers), so every dummy copy of group-m
#' variables is subject to the same cluster penalty as the originals.
#'
#' Key advantage: M << p_d, so the augmented system is far smaller than
#' the EN augmentation (M rows vs. p_d rows appended).
#'
#' @param X_tilde     Extended matrix [X | dummy_block], n x p_d.
#' @param y           Response vector, length n.
#' @param lambda_2_lars Ridge penalty parameter lambda_2 > 0.
#' @param gvs_setup   Output of .setup_gvs(). Uses:
#'                      $max_clusters      (= M)
#'                      $cluster_sizes     (= p_1,...,p_M)
#'                      $IEN_cl_id_vectors (= M x p binary support matrix)
#' @return list(X_aug, y_aug).
.augment_ien <- function(X_tilde, y, lambda_2_lars, gvs_setup) {
  p             <- ncol(gvs_setup$IEN_cl_id_vectors)   # original p
  p_d           <- ncol(X_tilde)                        # p + L
  M             <- gvs_setup$max_clusters
  p_m           <- gvs_setup$cluster_sizes              # vector length M
  one_m         <- gvs_setup$IEN_cl_id_vectors          # M x p

  # Replicate each support row (1 x p) across all (p_d / p) column blocks,
  # producing an (M x p_d) bottom block.
  bottom_block <- (1 / sqrt(p_m)) *
    matrix(
      rep(one_m, times = p_d / p),
      nrow = M,
      ncol = p_d
    )

  X_aug <- sqrt(lambda_2_lars) *
    rbind(
      (1 / sqrt(lambda_2_lars)) * X_tilde,
      bottom_block
    )
  y_aug <- append(y, rep(0, times = M))
  list(X_aug = X_aug, y_aug = y_aug)
}


# ==============================================================================
# .lm_dummy_gvs
# ==============================================================================

#' Run one random experiment of the T-Rex+GVS selector.
#'
#' Cold-start (T_stop == 1 or no model supplied):
#'   1. Draws X_tilde = [X | dummy_block] via .draw_dummies_gvs().
#'   2. Dispatches to the requested augmentation policy (.augment_en or
#'      .augment_ien) to obtain the Lasso-type system (X_aug, y_aug).
#'   3. Re-scales (X_aug, y_aug) and builds a new tlars_model.
#'
#' Warm-start (T_stop > 1 and model_tlars supplied):
#'   Advances the existing model by one T-LARS step. The augmented system
#'   is already stored inside model_tlars — no further GVS logic needed.
#'
#' @return Object of class tlars_cpp.
.lm_dummy_gvs <- function(X,
                          y,
                          model_tlars,
                          T_stop,
                          num_dummies,
                          GVS_type,
                          type,
                          gvs_setup,
                          lambda_2_lars,
                          early_stop  = TRUE,
                          intercept   = FALSE,
                          standardize = TRUE) {

  # ── Cold-start ────────────────────────────────────────────────────────────────
  if (T_stop == 1L || missing(model_tlars) || is.null(model_tlars)) {

    # Step 1: draw X_tilde = [X | dummy_block],  n x p_d
    X_tilde <- .draw_dummies_gvs(X, gvs_setup, num_dummies)

    # Step 2: augmentation policy dispatch → (X_aug, y_aug)
    aug <- switch(GVS_type,
      EN  = .augment_en( X_tilde, y, lambda_2_lars, gvs_setup),
      IEN = .augment_ien(X_tilde, y, lambda_2_lars, gvs_setup),
      stop("Unknown GVS_type '", GVS_type, "'. ",
           "Implement a corresponding .augment_*() policy and register it here.")
    )

    # Step 3: re-scale augmented system and build T-LARS model
    X_aug <- scale(aug$X_aug)
    y_aug <- aug$y_aug - mean(aug$y_aug)

    model_tlars <- tlars::tlars_model(
      X           = X_aug,
      y           = y_aug,
      num_dummies = num_dummies,
      verbose     = FALSE,
      info        = FALSE,
      intercept   = intercept,
      standardize = standardize,
      type        = type
    )
  }

  # ── Advance one T-LARS step (cold- or warm-start) ────────────────────────────
  tlars::tlars(
    model      = model_tlars,
    T_stop     = T_stop,
    early_stop = early_stop,
    info       = FALSE
  )

  return(model_tlars)
}


# ==============================================================================
# .random_exp_gvs
# ==============================================================================

#' Run K random experiments for the T-Rex+GVS selector.
#'
#' Orchestrates K calls to .lm_dummy_gvs(), collects the solution paths,
#' and returns the accumulated phi_T_mat and Phi vectors.
#'
#' @return Named list: phi_T_mat, Phi, lars_state_list, dummy_last_betas.
.random_exp_gvs <- function(X,
                            y,
                            K,
                            T_stop,
                            num_dummies,
                            GVS_type,
                            type,
                            gvs_setup,
                            lambda_2_lars,
                            early_stop         = TRUE,
                            lars_state_list    = NULL,
                            verbose            = TRUE,
                            intercept          = FALSE,
                            standardize        = TRUE,
                            dummy_coef         = FALSE,
                            parallel_process   = FALSE,
                            parallel_max_cores = 1L,
                            seed               = NULL,
                            eps                = .Machine$double.eps) {

  p <- ncol(X)

  if (is.null(lars_state_list))
    lars_state_list <- vector(mode = "list", length = K)

  # ── Combiner for parallel foreach output ─────────────────────────────────────
  .comb_fun <- function(x, ...) {
    lapply(seq_along(x), function(i)
      c(x[[i]], lapply(list(...), function(y) y[[i]])))
  }

  `%par_exe%` <- if (parallel_process) doRNG::`%dorng%` else foreach::`%do%`

  h <- NULL
  res <- foreach::foreach(
    h             = seq(K),
    .combine      = .comb_fun,
    .multicombine = TRUE,
    .init         = list(list(), list(), list(), list()),
    .options.RNG  = seed
  ) %par_exe% {

    # Recreate tlars_cpp object from serialised state (parallel transport)
    lars_state <-
      if (parallel_process &&
          !is.null(lars_state_list[[h]]) &&
          !methods::is(lars_state_list[[h]], tlars::tlars_cpp))
        tlars::tlars_model(lars_state = lars_state_list[[h]])
      else
        lars_state_list[[h]]

    # One random experiment
    lars_state <- .lm_dummy_gvs(
      X             = X,
      y             = y,
      model_tlars   = lars_state,
      T_stop        = T_stop,
      num_dummies   = num_dummies,
      GVS_type      = GVS_type,
      type          = type,
      gvs_setup     = gvs_setup,
      lambda_2_lars = lambda_2_lars,
      early_stop    = early_stop,
      intercept     = intercept,
      standardize   = standardize
    )

    # Extract full solution path
    lars_path <- do.call(cbind, lars_state$get_beta_path())

    # Serialise for parallel transport
    if (parallel_process)
      lars_state <- lars_state$get_all()

    # Number of included dummies at each step along the solution path
    dummy_num_path <- colSums(
      matrix(abs(lars_path[(p + 1L):(p + num_dummies), ]) > eps,
             nrow = num_dummies, ncol = ncol(lars_path))
    )

    # Relative occurrences: phi_T_mat[j, t] = (1/K) * 1{var j active at T = t}
    phi_T_mat <- matrix(0, nrow = p, ncol = T_stop)
    for (c in seq(T_stop)) {
      if (!any(dummy_num_path == c)) {
        ind_sol_path <- length(dummy_num_path)
        warning(paste0("T_stop = ", c, ": LARS reaches k = min(n,p) ",
                       "without selecting ", c, " dummies."))
      } else {
        ind_sol_path <- which(as.numeric(dummy_num_path) == c)[1L]
      }
      phi_T_mat[, c] <- (1 / K) * (abs(lars_path[1:p, ind_sol_path]) > eps)
    }

    last_betas <- lars_path[1:p, ncol(lars_path)]
    dummy_last <- if (dummy_coef)
      lars_path[seq(p + 1L, p + num_dummies), ncol(lars_path)]
    else
      NULL

    list(phi_T_mat, last_betas, lars_state, dummy_last)
  }

  # ── Merge K results ───────────────────────────────────────────────────────────
  lars_state_list        <- res[[3]]
  names(lars_state_list) <- paste0("lars_state (K = ", seq(K), ")")

  phi_T_mat        <- Reduce("+",   res[[1]])
  last_betas_mat   <- unname(Reduce(rbind, res[[2]]))
  Phi              <- apply(abs(last_betas_mat) > eps, 2, sum) / K
  dummy_last_betas <- if (dummy_coef) unname(Reduce(rbind, res[[4]])) else NULL

  list(
    phi_T_mat        = phi_T_mat,
    Phi              = Phi,
    lars_state_list  = lars_state_list,
    dummy_last_betas = dummy_last_betas
  )
}


# ==============================================================================
# trex_GVS  (main entry point)
# ==============================================================================

#' Run the T-Rex+GVS selector
#'
#' Performs fast FDR-controlled variable selection in high-dimensional
#' settings by generating cluster-aware dummy predictors. The group
#' structure may be supplied as prior knowledge or estimated automatically
#' from X via single-linkage hierarchical clustering.
#'
#' @param X               Real-valued predictor matrix (n x p).
#' @param y               Response vector (length n).
#' @param tFDR            Target FDR level in [0, 1]. Default 0.2.
#' @param K               Number of random experiments (integer >= 2).
#'                        Default 20.
#' @param max_num_dummies Integer multiplier; max dummies = factor * p.
#'                        Default 10.
#' @param max_T_stop      If TRUE, max T_stop = ceiling(n / 2). Default TRUE.
#' @param GVS_type        Augmentation policy. Currently supported:
#'                        'IEN' — Informed Elastic Net
#'                        (\doi{10.1109/CAMSAP58249.2023.10403489}),
#'                        'EN'  — ordinary Elastic Net
#'                        (\doi{10.1111/j.1467-9868.2005.00503.x}).
#'                        Default 'IEN'. Future Tikhonov-type policies
#'                        are registered here and in .lm_dummy_gvs().
#' @param type            'lar' (default) or 'lasso'.
#'
#' @param groups          Prior group knowledge: integer vector of length p
#'                        where groups[j] is the cluster index of variable j.
#'                        When supplied, corr_max is ignored and no
#'                        hierarchical clustering is performed.
#'                        Default NULL (auto-compute from X).
#' @param group_labels    Optional character vector of length
#'                        length(unique(groups)), providing human-readable
#'                        names for each cluster. Only used when groups != NULL.
#'                        Default NULL.
#'
#' @param corr_max        Maximum allowed pairwise correlation between
#'                        variables from different clusters (used only when
#'                        groups = NULL). Default 0.5.
#' @param hc_method       Linkage method for hierarchical clustering (used only
#'                        when groups = NULL). Exactly one of:
#'                        'single'   (default) — consistent with Machkour et al.
#'                          (CAMSAP 2023); one pair per cluster has |cor| >= corr_max,
#'                        'complete' — all pairs in cluster have |cor| >= corr_max;
#'                          tightest guarantee on within-cluster correlation,
#'                        'average'  — UPGMA; average pairwise |cor| >= corr_max,
#'                        'mcquitty' — WPGMA; weighted average linkage.
#'                        Any other value, including 'ward.D2', raises an error.
#'                        Note on 'ward.D2': intentionally not supported.
#'                        Ward's criterion minimises the total within-cluster
#'                        sum of squared Euclidean distances. It is only
#'                        mathematically well-defined for Euclidean distance
#'                        matrices. The distance measure used here is
#'                        d = 1 - |cor(x_j, x_j')|, which is not a Euclidean
#'                        distance. Passing 'ward.D2' with a correlation-based
#'                        distance matrix silently produces geometrically
#'                        meaningless clusters and invalidates the corr_max
#'                        threshold guarantee. If Ward clustering on X is
#'                        genuinely needed, compute the partition externally
#'                        (e.g. hclust(dist(t(X)), method = "ward.D2")) and
#'                        supply the result via the 'groups' argument.
#' @param lambda_2_lars   lambda_2-value for the LARS-based Elastic Net.
#'                        Computed automatically via 10-fold cross-validated
#'                        ridge if NULL (default).
#'
#' @param parallel_process    Logical. Default FALSE.
#' @param parallel_max_cores  Maximum cores for parallel execution.
#' @param seed            RNG seed (ignored when parallel_process = FALSE).
#' @param eps             Numerical zero. Default .Machine$double.eps.
#' @param verbose         Print progress. Default TRUE.
#'
#' @return Named list:
#'   \item{selected_var}{Binary vector of length p.}
#'   \item{tFDR, T_stop, num_dummies, V, v_thresh}{Calibration outputs.}
#'   \item{FDP_hat_mat, Phi_mat, R_mat}{Accumulated calibration matrices.}
#'   \item{phi_T_mat, Phi_prime}{Terminal relative occurrences.}
#'   \item{GVS_type, type, corr_max, lambda_2_lars}{Parameters used.}
#'   \item{groups, group_labels}{Group structure used.}
#'
#' @importFrom parallel detectCores makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach getDoParWorkers registerDoSEQ
#' @importFrom stats cov cov2cor as.dist hclust cutree
#' @importFrom methods is
#' @export
#'
#' @examples
#' set.seed(1234)
#' n <- 50; p <- 100
#' X <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
#' beta <- c(rep(3, 3), rep(0, 97))
#' y <- X %*% beta + stats::rnorm(n)
#'
#' # Auto-computed clustering
#' res <- trex_GVS(X = X, y = y, K = 20)
#'
#' # Prior group knowledge
#' grps <- rep(1:10, each = 10)
#' res2 <- trex_GVS(X = X, y = y, groups = grps,
#'                  group_labels = paste0("G", 1:10), K = 20)
trex_GVS <- function(X,
                     y,
                     tFDR                = 0.2,
                     K                   = 20,
                     max_num_dummies     = 10,
                     max_T_stop          = TRUE,
                     GVS_type            = "IEN",
                     type                = "lar",
                     groups              = NULL,
                     group_labels        = NULL,
                     corr_max            = 0.5,
                     hc_method           = "single",
                     lambda_2_lars       = NULL,
                     parallel_process    = FALSE,
                     parallel_max_cores  = min(K, max(1L, parallel::detectCores(logical = FALSE))),
                     seed                = NULL,
                     eps                 = .Machine$double.eps,
                     verbose             = TRUE) {

  # ── Argument matching ────────────────────────────────────────────────────────
  GVS_type <- match.arg(GVS_type, c("IEN", "EN"))
  type     <- match.arg(type,     c("lar", "lasso"))
  hc_method <- match.arg(hc_method,
                         c("single", "complete", "average", "mcquitty"))


  # ── Input validation ─────────────────────────────────────────────────────────
  .check_trex_gvs(X, y, tFDR, K, max_num_dummies,
                  groups, group_labels,
                  corr_max, hc_method, lambda_2_lars,
                  parallel_process, parallel_max_cores, eps)

  # Silently clamp parallel_max_cores to physical limit
  if (parallel_process) {
    hard_limit <- min(K, max(1L, parallel::detectCores(logical = FALSE)))
    if (parallel_max_cores > hard_limit) {
      message("Setting parallel_max_cores = ", hard_limit, " (physical cores).")
      parallel_max_cores <- hard_limit
    }
  }

  # ── Dimensions and preprocessing ─────────────────────────────────────────────
  n <- nrow(X)
  p <- ncol(X)
  if (p > 1e4) warning("Feature space > 10,000 variables. RAM issues may occur.")

  X <- scale(X)
  y <- y - mean(y)

  # ── Group structure: computed ONCE ───────────────────────────────────────────
  gvs_setup <- .setup_gvs(X, corr_max, hc_method, groups, group_labels)

  # ── lambda_2: computed ONCE ───────────────────────────────────────────────────
  if (is.null(lambda_2_lars))
    lambda_2_lars <- .compute_lambda2(X, y,
                                      intercept   = FALSE,
                                      standardize = TRUE)

  # ── Voting grid ───────────────────────────────────────────────────────────────
  V         <- seq(0.5, 1 - eps, by = 1 / K)
  V_len     <- length(V)
  opt_point <- which(abs(V - 0.75) < eps)
  if (length(opt_point) == 0)
    opt_point <- length(V[V < 0.75])

  # ── Parallel cluster ──────────────────────────────────────────────────────────
  if (parallel_process && foreach::getDoParWorkers() == 1) {
    cl <- parallel::makeCluster(parallel_max_cores)
    doParallel::registerDoParallel(cl)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    on.exit(foreach::registerDoSEQ(),  add = TRUE)
  }

  # ── Step closure: K experiments + Phi_prime + FDP_hat ────────────────────────
  .run_step <- function(T_stop, num_dummies, lars_state_list = NULL) {
    suppressWarnings(
      rand_exp <- .random_exp_gvs(
        X                  = X,
        y                  = y,
        K                  = K,
        T_stop             = T_stop,
        num_dummies        = num_dummies,
        GVS_type           = GVS_type,
        type               = type,
        gvs_setup          = gvs_setup,
        lambda_2_lars      = lambda_2_lars,
        early_stop         = TRUE,
        lars_state_list    = lars_state_list,
        verbose            = verbose,
        intercept          = FALSE,
        standardize        = TRUE,
        parallel_process   = parallel_process,
        parallel_max_cores = parallel_max_cores,
        seed               = seed,
        eps                = eps
      )
    )

    Phi_prime <- Phi_prime_fun(
      p           = p,
      T_stop      = T_stop,
      num_dummies = num_dummies,
      phi_T_mat   = rand_exp$phi_T_mat,
      Phi         = rand_exp$Phi,
      eps         = eps
    )

    FDP_hat <- fdp_hat(V = V, Phi = rand_exp$Phi, Phi_prime = Phi_prime)

    list(rand_exp  = rand_exp,
         phi_T_mat = rand_exp$phi_T_mat,
         Phi       = rand_exp$Phi,
         Phi_prime = Phi_prime,
         FDP_hat   = FDP_hat)
  }

  # ── L-loop: grow num_dummies until FDP_hat <= tFDR at 75% voting level ───────
  LL      <- 1L
  T_stop  <- 1L
  FDP_hat <- rep(NA_real_, times = V_len)

  while ((LL <= max_num_dummies &&
          (is.na(FDP_hat[opt_point]) || FDP_hat[opt_point] > tFDR)) ||
         sum(!is.na(FDP_hat)) == 0) {
    num_dummies <- LL * p
    LL          <- LL + 1L
    step        <- .run_step(T_stop, num_dummies)
    FDP_hat     <- step$FDP_hat
    if (verbose) cat("\n Appended dummies:", num_dummies, "\n")
  }

  # ── T-loop: extend T_stop while FDP fits under tFDR ──────────────────────────
  acc <- list(FDP = matrix(FDP_hat, nrow = 1),
              Phi = matrix(step$Phi, nrow = 1))

  max_T <- if (max_T_stop) min(num_dummies, ceiling(n / 2)) else num_dummies
  if (!is.null(seed)) seed <- seed + 12345L

  while (FDP_hat[V_len] <= tFDR && T_stop < max_T) {
    T_stop  <- T_stop + 1L
    step    <- .run_step(T_stop, num_dummies, step$rand_exp$lars_state_list)
    FDP_hat <- step$FDP_hat
    acc     <- list(FDP = rbind(acc$FDP, FDP_hat),
                    Phi = rbind(acc$Phi, step$Phi))
  }

  if (verbose) cat("\n Included dummies before stopping:", T_stop, "\n")

  # ── Variable selection ────────────────────────────────────────────────────────
  sel <- select_var_fun(p           = p,
                        tFDR        = tFDR,
                        T_stop      = T_stop,
                        FDP_hat_mat = acc$FDP,
                        Phi_mat     = acc$Phi,
                        V           = V)

  # ── Return ────────────────────────────────────────────────────────────────────
  list(
    selected_var  = sel$selected_var,
    tFDR          = tFDR,
    T_stop        = T_stop,
    num_dummies   = num_dummies,
    V             = V,
    v_thresh      = sel$v_thresh,
    FDP_hat_mat   = acc$FDP,
    Phi_mat       = acc$Phi,
    R_mat         = sel$R_mat,
    phi_T_mat     = step$phi_T_mat,
    Phi_prime     = step$Phi_prime,
    GVS_type      = GVS_type,
    type          = type,
    corr_max      = corr_max,
    hc_method     = gvs_setup$hc_method,
    lambda_2_lars = lambda_2_lars,
    groups        = groups,
    group_labels  = gvs_setup$group_labels
  )
}
