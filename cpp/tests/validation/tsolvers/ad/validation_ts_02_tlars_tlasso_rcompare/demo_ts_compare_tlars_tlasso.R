# =============================================================================
# demo_ts_compare_tlars_tlasso.R
# =============================================================================
#
# Purpose
# -------
# Reference generator for a forward-selection path-equivalence test of the CRAN
# `tlars` package against the C++ tsolvers solvers, now driven by the SAME sparse
# factor-model DGP used for the SPCA evaluation (demo_trex_spca_01/02.R) with
# explicit SNR control.  This is the "final" solver test: realistic, strongly
# correlated columns (factor structure + an overlap pool) -- the conditions that
# actually trigger LARS ties and collinearity/LASSO drops -- rather than the
# near-orthogonal Gaussian data of the first iteration.
#
# Four solvers are exercised:
#   - LARS   : CRAN tlars type="lar"   vs C++ TLARS_Solver
#   - LASSO  : CRAN tlars type="lasso" vs C++ TLASSO_Solver
#   - EN     : CRAN augmented-lasso (lasso_star recipe) vs C++ TENET_Solver
#              (Gram-based EN) AND TENETAug_Solver (augmented-lasso EN, default
#              TLASSO inner).  Both EN solvers perform LASSO-style sign-change
#              drops, so the matching R reference is the augmented type="lasso".
#
# API mapping (the crucial point)
# -------------------------------
#   - CRAN tlars takes ONE augmented matrix  Z = cbind(X, D)  (dim n x (p+L))
#     plus num_dummies = L.  First p columns = true predictors, last L = dummies.
#   - The C++ solvers take X (n x p) and D (n x L) SEPARATELY and concatenate
#     internally.  R column j (1-based) <-> C++ combined 0-based index (j-1).
#
# Elastic-net augmentation (lasso_star, == TENETAug_Solver):
#   d1 = sqrt(lambda2), d2 = 1/sqrt(1+lambda2)
#   Zaug = d2 * rbind(Z, d1 * I_{p+L})   (column L2-norms become exactly 1)
#   yaug = c(y, 0_{p+L})
#   beta_original = beta_star / d2        (== TENET/TENETAug getBetaPath())
#
# To remove every preprocessing confound, Z is pre-standardised (centre +
# unit-L2 per column) and y centred ONCE; the processed Xn/Dn/y are dumped
# verbatim and fed to both sides with standardize=FALSE / normalize=false and
# the COMPLETE path (early_stop = FALSE).  Both solvers therefore see
# BIT-IDENTICAL inputs.
#
# Outputs (./rdump_tlars/ next to this script):
#   meta.csv                         single row: n,p,L,num,lambda2,snr_db
#   Xn_<i>.csv (n x p), Dn_<i>.csv (n x L), y_<i>.csv (n x 1)
#   r_lar_beta_<i>.csv   / r_lar_act_<i>.csv     : LARS  reference path + actions
#   r_lasso_beta_<i>.csv / r_lasso_act_<i>.csv   : LASSO reference path + actions
#   r_en_beta_<i>.csv    / r_en_act_<i>.csv      : EN (augmented lasso) reference
#
# Companion: the co-located validation_ts_02_tlars_tlasso_rcompare.cpp gate.
# =============================================================================

suppressMessages(library(tlars))
suppressMessages(library(TRexSelector))   # for the GVS dummy-from-clustering reference

# ---- locate this script's directory (works under Rscript and source()) ------
get_script_dir <- function() {
  a <- commandArgs(FALSE)
  f <- grep("^--file=", a, value = TRUE)
  if (length(f)) return(dirname(normalizePath(sub("^--file=", "", f[1]))))
  if (!is.null(sys.frames()[[1]]$ofile))
    return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  getwd()
}

# ---- full-precision (round-trip) CSV writer ---------------------------------
write_mat <- function(M, path) {
  M <- as.matrix(M)
  con <- file(path, "w")
  on.exit(close(con))
  for (i in seq_len(nrow(M))) {
    cat(paste(formatC(M[i, ], format = "e", digits = 17), collapse = ","),
        "\n", file = con, sep = "")
  }
}

# ---- compact integer-matrix writer (cluster labels) -------------------------
write_int_mat <- function(M, path) {
  M <- as.matrix(M)
  con <- file(path, "w")
  on.exit(close(con))
  for (i in seq_len(nrow(M))) {
    cat(paste(as.integer(M[i, ]), collapse = ","), "\n", file = con, sep = "")
  }
}

# =============================================================================
# Sparse factor-model DGP  --  byte-for-byte identical to demo_trex_spca_01/02.R
# =============================================================================
dgp_sparse_factor_model <- function(n, p, p1, M, target_snr_db, overlap_pool) {
  Z <- matrix(0, nrow = n, ncol = M)
  factor_stds <- c(5.0, 3.0, 1.0)
  for (m in 1:M) {
    std_dev <- ifelse(m <= length(factor_stds), factor_stds[m], 1.0)
    Z[, m] <- rnorm(n, mean = 0, sd = std_dev)
  }

  V <- matrix(0, nrow = p, ncol = M)
  pool_indices <- 1:min(overlap_pool, p)
  for (m in 1:M) {
    active_idx <- sample(pool_indices, p1, replace = FALSE)
    V[active_idx, m] <- 0.9
  }

  S          <- Z %*% t(V)
  signal_var <- var(as.vector(S))

  E_raw         <- matrix(rnorm(n * p), nrow = n, ncol = p)
  raw_noise_var <- var(as.vector(E_raw))
  target_noise_var <- signal_var / (10^(target_snr_db / 10))
  E <- E_raw * sqrt(target_noise_var / raw_noise_var)

  X <- S + E
  X <- scale(X, center = TRUE, scale = FALSE)
  list(X = X, V = V, Z = Z)
}

# =============================================================================
# Configuration
# =============================================================================
set.seed(2025)

n            <- 60     # observations
p            <- 40     # predictors
L            <- 40     # dummies
p1           <- 5      # true support size (subset of the overlap pool)
M            <- 3      # number of latent factors
overlap_pool <- 20     # strongly-correlated index pool (ties/collinearity)
snr_db       <- -7     # SNR control knob (dB); SPCA worst-gap operating point
lambda2      <- 0.5    # elastic-net ridge parameter (LARS units) for TENET tests
num          <- 10     # number of independent datasets
seed_base    <- 42

# corr_max grid for the GVS dummy-from-clustering check.  The GVS dummy matrix
# is built on the clusters obtained by cutting the single-linkage dendrogram at
# height h = 1 - corr_max (see TRexSelector:::add_dummies_GVS).  A spread of
# corr_max exercises the whole range from many singletons (low corr_max -> tall
# cut) to one big cluster (high corr_max -> shallow cut).
corr_max_vec <- c(0.20, 0.35, 0.50, 0.70)

# Regenerate the committed reference dumps in the shared library data folder,
# two levels up from this validation folder (<validation>/data/rdump_tlars),
# which is where the cpp gate and the Python mirror read them from.
outdir <- normalizePath(file.path(get_script_dir(), "..", "..", "data", "rdump_tlars"),
                        mustWork = FALSE)
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

write_mat(matrix(c(n, p, L, num, lambda2, snr_db), nrow = 1),
          file.path(outdir, "meta.csv"))
write_mat(matrix(corr_max_vec, nrow = 1),
          file.path(outdir, "gvs_corrmax.csv"))

cat(sprintf("tlars factor-model path-equivalence reference\n"))
cat(sprintf("  n=%d p=%d L=%d p1=%d M=%d overlap=%d SNR=%ddB lambda2=%.3f num=%d\n",
            n, p, L, p1, M, overlap_pool, snr_db, lambda2, num))
cat(sprintf("  outdir = %s\n\n", outdir))

# ---- elastic-net augmentation constants -------------------------------------
d1 <- sqrt(lambda2)
d2 <- 1 / sqrt(1 + lambda2)

# ---- run one tlars configuration to its complete path -----------------------
run_plain <- function(Z, y, L, type) {
  m <- suppressMessages(tlars_model(
    X = Z, y = y, num_dummies = L,
    verbose = FALSE, intercept = FALSE, standardize = FALSE,
    type = type, info = FALSE))
  suppressMessages(tlars(m, T_stop = L, early_stop = FALSE, info = FALSE))
  list(beta_path = t(do.call(rbind, m$get_beta_path())),  # (p+L) x (steps+1)
       actions   = unlist(m$get_actions()))
}

# Our own augmented elastic-net path, built on the CRAN tlars TLARS/TLASSO core:
#   type="lasso" -> LARS-LASSO inner (sign-change drops) == TENET / default TENET_AUG
#   type="lar"   -> pure-LARS inner (never drops)        == TENET_AUG(use_lars_inner=TRUE)
run_en <- function(Z, y, L, d1, d2, type) {
  pL   <- ncol(Z)
  Zaug <- d2 * rbind(Z, d1 * diag(pL))      # (n+pL) x pL ; unit-L2 columns
  yaug <- c(y, rep(0, pL))
  m <- suppressMessages(tlars_model(
    X = Zaug, y = yaug, num_dummies = L,
    verbose = FALSE, intercept = FALSE, standardize = FALSE,
    type = type, info = FALSE))
  suppressMessages(tlars(m, T_stop = L, early_stop = FALSE, info = FALSE))
  bp <- do.call(rbind, m$get_beta_path()) / d2  # (steps+1) x pL, original scale
  list(beta_path = t(bp), actions = unlist(m$get_actions()))
}

# Single-linkage hierarchical clustering of the p columns of Xn on the
# correlation distance 1 - |cor| -- the SAME metric/linkage the C++
# AgglomerativeClustering<Correlation, Single> uses in TRex-DA.
run_clustering <- function(Xn) {
  pp  <- ncol(Xn)
  Dst <- 1 - abs(stats::cor(Xn))            # p x p ; diag 0
  hc  <- stats::hclust(stats::as.dist(Dst), method = "single")
  # per-K partition labels (row k = cutree to k clusters), K = 1..p
  labs <- t(vapply(seq_len(pp), function(k) stats::cutree(hc, k = k),
                   integer(pp)))            # p x p integer matrix
  list(height = sort(hc$height), labels = labs)
}

# GVS dummy-from-clustering reference.  Drives the REAL production function
# TRexSelector:::add_dummies_GVS(), which:
#   1. clusters the columns via hclust(as.dist(1-|cov2cor(cov(X))|), "single"),
#   2. cuts the dendrogram at height h = 1 - corr_max  (cutree(h = ...)),
#   3. draws each cluster's dummy block from MASS::mvrnorm(n, 0, cov(sub_X)).
# Steps (1)-(2) fix the cluster PARTITION; step (3) fixes the per-cluster
# generating COVARIANCE.  Both are deterministic given X and corr_max (the
# mvrnorm draws are not, so we compare the distribution, not the samples):
#   labels    : per-variable 1-based cluster id (length p)
#   Sigma_gen : p x p block-diagonal cov(X) -- within-cluster blocks = cov(sub_X),
#               zero across clusters -- i.e. the exact covariance the dummies are
#               sampled from.  This is what the C++ GVS setup must reproduce.
run_gvs_dummy_ref <- function(Xn, corr_max) {
  pp  <- ncol(Xn)
  res <- TRexSelector:::add_dummies_GVS(X = Xn, num_dummies = pp,
                                        corr_max = corr_max)
  idv <- res$IEN_cl_id_vectors                 # max_clusters x p boolean
  if (is.null(dim(idv))) idv <- matrix(idv, nrow = 1)
  labels  <- apply(idv, 2, function(col) which(col)[1])   # 1-based, length p
  Sig     <- stats::cov(Xn)                     # == cov(sub_X) submatrices
  same    <- outer(labels, labels, FUN = "==")
  Sig_gen <- Sig * same                         # block-diagonal generating cov
  list(labels = labels, max_clusters = res$max_clusters, Sigma_gen = Sig_gen)
}

# =============================================================================
# Main loop
# =============================================================================
for (i in seq_len(num) - 1L) {
  set.seed(seed_base + i * 1000 + snr_db)
  dgp <- dgp_sparse_factor_model(n, p, p1, M, snr_db, overlap_pool)
  X   <- dgp$X

  # PC1 score from ordinary PCA on the (centred) design -- the SPCA response
  sv  <- svd(X)
  y   <- as.numeric(X %*% sv$v[, 1])

  # dummies: independent standard normal noise columns
  D   <- matrix(rnorm(n * L), n, L)

  # shared preprocessing: centre + unit-L2 per column on Z; centre y
  Z   <- cbind(X, D)
  Z   <- scale(Z, center = TRUE, scale = FALSE)
  nrm <- sqrt(colSums(Z^2)); nrm[nrm == 0] <- 1
  Z   <- sweep(Z, 2, nrm, "/")
  y   <- y - mean(y)

  Xn <- Z[, 1:p, drop = FALSE]
  Dn <- Z[, (p + 1):(p + L), drop = FALSE]

  write_mat(Xn, file.path(outdir, sprintf("Xn_%d.csv", i)))
  write_mat(Dn, file.path(outdir, sprintf("Dn_%d.csv", i)))
  write_mat(matrix(y, ncol = 1), file.path(outdir, sprintf("y_%d.csv", i)))

  lar     <- run_plain(Z, y, L, "lar")
  lasso   <- run_plain(Z, y, L, "lasso")
  en_la   <- run_en(Z, y, L, d1, d2, "lasso")   # LASSO-inner augmented EN
  en_lar  <- run_en(Z, y, L, d1, d2, "lar")     # LARS-inner  augmented EN
  clust   <- run_clustering(Xn)

  write_mat(lar$beta_path,    file.path(outdir, sprintf("r_lar_beta_%d.csv", i)))
  write_mat(lasso$beta_path,  file.path(outdir, sprintf("r_lasso_beta_%d.csv", i)))
  write_mat(en_la$beta_path,  file.path(outdir, sprintf("r_en_beta_%d.csv", i)))
  write_mat(en_lar$beta_path, file.path(outdir, sprintf("r_enlar_beta_%d.csv", i)))
  cat(paste(lar$actions,    collapse = ","), "\n",
      file = file.path(outdir, sprintf("r_lar_act_%d.csv", i)), sep = "")
  cat(paste(lasso$actions,  collapse = ","), "\n",
      file = file.path(outdir, sprintf("r_lasso_act_%d.csv", i)), sep = "")
  cat(paste(en_la$actions,  collapse = ","), "\n",
      file = file.path(outdir, sprintf("r_en_act_%d.csv", i)), sep = "")
  cat(paste(en_lar$actions, collapse = ","), "\n",
      file = file.path(outdir, sprintf("r_enlar_act_%d.csv", i)), sep = "")

  # clustering reference
  write_mat(matrix(clust$height, ncol = 1),
            file.path(outdir, sprintf("r_clust_height_%d.csv", i)))
  write_int_mat(clust$labels,
            file.path(outdir, sprintf("r_clust_labels_%d.csv", i)))

  # GVS dummy-from-clustering reference, one set per corr_max cut point
  for (cm in corr_max_vec) {
    g   <- run_gvs_dummy_ref(Xn, cm)
    tag <- sprintf("%03d", as.integer(round(cm * 100)))
    write_int_mat(matrix(g$labels, nrow = 1),
                  file.path(outdir, sprintf("r_gvs_labels_%d_%s.csv", i, tag)))
    write_mat(g$Sigma_gen,
                  file.path(outdir, sprintf("r_gvs_sigma_%d_%s.csv", i, tag)))
  }

  cat(sprintf("  dataset %2d: lar=%d lasso=%d en(lasso)=%d en(lar)=%d  clust ok\n",
              i, length(lar$actions), length(lasso$actions),
              length(en_la$actions), length(en_lar$actions)))
}

cat("\nDone. Build + run the C++ comparator:\n")
cat("  validation_ts_02_tlars_tlasso_rcompare --dir ", outdir, "\n", sep = "")
