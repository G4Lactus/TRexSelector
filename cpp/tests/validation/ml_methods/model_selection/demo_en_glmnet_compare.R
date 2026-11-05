# =============================================================================
# demo_en_glmnet_compare.R
# =============================================================================
#
# Reference generator for the C++ coordinate-descent elastic-net solver
#   trex::ml_methods::model_selection::elastic_net_gaussian      (path)
#   trex::ml_methods::model_selection::elastic_net_cv_gaussian   (K-fold CV)
# versus R `glmnet` / `cv.glmnet`, on the SAME sparse factor-model design used
# for the SPCA / tsolver evaluations.
#
# Why this is the "best evidence" for lambda_2:
#   glmnet IS cyclic coordinate descent.  The C++ CD engine therefore matches
#   glmnet's coefficient PATH to solver tolerance (not merely within CV noise),
#   unlike the SVD-based ridge_cv / ridge_gcv / ridge_cv_glmnet / ridge_cv_direct
#   which only approximate cv.glmnet's lambda.min / lambda.1se.
#
# Dumps (./rdump_en next to this script):
#   meta.csv                     : n, p   (single row)
#   Xn.csv (n x p), y.csv (n x 1): centered factor design + PC1 score
#   For each alpha in {0.0, 0.5, 1.0}  (tag 000 / 050 / 100):
#     glmnet_lambda_<tag>.csv    : fit$lambda (descending), 1 column
#     glmnet_beta_<tag>.csv      : p x nlambda coefficient path (original scale)
#     glmnet_a0_<tag>.csv        : 1 x nlambda intercepts
#   cv_glmnet_<tag>.csv          : lambda.min, lambda.1se  (mean over reps), 1 row
#
# Companion: cpp/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd/demo_mlm_ms_02_enet_cv_ccd.cpp
# =============================================================================

suppressPackageStartupMessages(library(glmnet))

# ---- locate this script's directory (Rscript and source()) ------------------
get_script_dir <- function() {
  a <- commandArgs(FALSE)
  f <- grep("^--file=", a, value = TRUE)
  if (length(f)) return(dirname(normalizePath(sub("^--file=", "", f[1]))))
  if (!is.null(sys.frames()[[1]]$ofile))
    return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  getwd()
}

# ---- full-precision matrix writer (e-format, 17 digits) ---------------------
write_mat <- function(M, path) {
  M <- as.matrix(M)
  con <- file(path, "w"); on.exit(close(con))
  for (i in seq_len(nrow(M)))
    cat(paste(formatC(M[i, ], format = "e", digits = 17), collapse = ","),
        "\n", file = con, sep = "")
}

# =============================================================================
# Sparse factor-model DGP  (identical to demo_ts_compare_tlars_tlasso.R)
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
  E_raw      <- matrix(rnorm(n * p), nrow = n, ncol = p)
  raw_noise_var    <- var(as.vector(E_raw))
  target_noise_var <- signal_var / (10^(target_snr_db / 10))
  E <- E_raw * sqrt(target_noise_var / raw_noise_var)
  X <- S + E
  X <- scale(X, center = TRUE, scale = FALSE)
  list(X = X)
}

# =============================================================================
# Configuration
# =============================================================================
set.seed(2025)
n <- 60; p <- 40; p1 <- 5; M <- 3; overlap_pool <- 20; snr_db <- -7
alphas <- c(0.0, 0.5, 1.0)
n_lambda <- 100
n_cv_rep <- 25L

outdir <- file.path(get_script_dir(), "rdump_en")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

dgp <- dgp_sparse_factor_model(n, p, p1, M, snr_db, overlap_pool)
X   <- dgp$X
sv  <- svd(X)
y   <- as.numeric(X %*% sv$v[, 1])   # PC1 score (centered)

write_mat(matrix(c(n, p), nrow = 1), file.path(outdir, "meta.csv"))
write_mat(X,                         file.path(outdir, "Xn.csv"))
write_mat(matrix(y, ncol = 1),       file.path(outdir, "y.csv"))

cat(sprintf("glmnet <-> C++ coordinate-descent EN reference\n"))
cat(sprintf("  n=%d p=%d  alphas={%s}  n_lambda=%d  glmnet=%s\n",
            n, p, paste(alphas, collapse = ","), n_lambda,
            as.character(packageVersion("glmnet"))))
cat(sprintf("  outdir = %s\n\n", outdir))

tag_of <- function(a) sprintf("%03d", as.integer(round(a * 100)))

for (a in alphas) {
  tag <- tag_of(a)

  # full path at a fixed grid length; standardize+intercept = glmnet defaults.
  # thresh tightened far below glmnet's 1e-7 default so the coefficient path is
  # solved to near machine precision -> isolates the CD algorithm from glmnet's
  # convergence slack for the C++ comparison.
  fit <- glmnet(x = X, y = y, family = "gaussian", alpha = a,
                standardize = TRUE, intercept = TRUE, nlambda = n_lambda,
                thresh = 1e-14)

  cf  <- as.matrix(coef(fit))        # (p+1) x nlam ; row 1 = intercept
  a0  <- cf[1, , drop = FALSE]       # 1 x nlam
  bet <- cf[-1, , drop = FALSE]      # p x nlam

  write_mat(matrix(fit$lambda, ncol = 1),
            file.path(outdir, sprintf("glmnet_lambda_%s.csv", tag)))
  write_mat(bet, file.path(outdir, sprintf("glmnet_beta_%s.csv", tag)))
  write_mat(a0,  file.path(outdir, sprintf("glmnet_a0_%s.csv", tag)))

  # repeated cv.glmnet (random folds) -> mean lambda.min / lambda.1se
  lmin <- numeric(n_cv_rep); l1se <- numeric(n_cv_rep)
  for (r in seq_len(n_cv_rep)) {
    set.seed(r)
    cvf <- cv.glmnet(x = X, y = y, family = "gaussian", alpha = a,
                     standardize = TRUE, intercept = TRUE,
                     type.measure = "mse", nfolds = 10)
    lmin[r] <- cvf$lambda.min
    l1se[r] <- cvf$lambda.1se
  }
  write_mat(matrix(c(mean(lmin), mean(l1se)), nrow = 1),
            file.path(outdir, sprintf("cv_glmnet_%s.csv", tag)))

  cat(sprintf(paste0("  alpha=%.2f  nlam=%3d  lam[min..max]=[%.4g, %.4g]  ",
                     "cv.min=%.5g cv.1se=%.5g\n"),
              a, length(fit$lambda), min(fit$lambda), max(fit$lambda),
              mean(lmin), mean(l1se)))
}

cat("\nDone.  Build + run the C++ comparator:\n")
cat("  demo_mlm_ms_02_enet_cv_ccd\n")
