# ==============================================================================
# demo_trex_spca_02.R  --  R REFERENCE DUMP for the C++ head-to-head
# ==============================================================================
#
# PURPOSE
# -------
# The residual C++-vs-R FDR gap (~3%) on the SPCA benchmark must be explained on
# IDENTICAL data, with the data-generating RNG and the CV-fold RNG removed as
# confounds. Because the ground truth (the true PC1 support) is only known to the
# generator, R is made the single source of truth here:
#
#   1. generate each dataset with the EXACT DGP of demo_trex_spca_01.R,
#   2. dump X at full double precision (so C++ reads bit-identical inputs),
#   3. run the EXACT "T-Rex EN" PC1 pipeline of demo_trex_spca_01.R,
#   4. dump truth, lambda_2_lars, the R selection, and the R PC1 scores.
#
# The C++ side (validation_trex_spca_04_rdump_pipeline) then loads these same X files
# and, with --use-r-lambda2, reuses R's lambda_2 per trial so the ONLY remaining
# difference is the T-Rex selector itself.
#
# OUTPUT FILES (written into ./rdump next to this script):
#   X_<mc>.csv     n x p centered design, comma-separated, NO header (full precision)
#   truth.csv      "mc,indices"        PC1 true support, dash-joined, 0-based
#   r_lambda2.csv  "mc,lambda2"        R lambda_2_lars per trial
#   r_results.csv  "mc,k,fdr,tpr,indices"   R selection (indices dash-joined, 0-based)
#   r_pc1.csv      mc, then n PC1 scores (full precision) -- to verify PCA match
#
# NOTE: this OVERWRITES the existing X_*.csv in rdump/ (previously C++-generated).
#       After running this, RE-RUN the C++ demo so cpp_pipeline.csv reflects R's X:
#         validation_trex_spca_04_rdump_pipeline --use-r-lambda2 --n <num_MC>
#
# MIGRATED: the T-Rex+GVS(EN) selection now runs through the TRexSelectorNeo R
# binding (TRexGVSSelector) instead of the CRAN TRexSelector::trex(). The legacy
# type = "lar"/"lasso" inner-solver switch maps to the EN solver via
# trex_gvs_control(en_solver = "TENET" / "TENET_AUG") -- see note at the calls.
# ==============================================================================

library(TRexSelectorNeo)
library(glmnet)

# ------------------------------------------------------------------------------
# Resolve this script's directory (works under Rscript and source())
# ------------------------------------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  }
  getwd()
}

# ------------------------------------------------------------------------------
# Configuration  (mirror demo_trex_spca_01.R; single SNR for a focused diff)
# ------------------------------------------------------------------------------
n            <- 50
p            <- 100
p1           <- 5
M            <- 3
overlap_pool <- 30
target_fdr   <- 0.10
num_MC       <- 100
snr_db       <- as.numeric(Sys.getenv("RDUMP_SNR_DB", "-7"))  # worst-gap operating point; override via env
seed_base    <- 42      # set.seed(seed_base + mc*1000 + snr_db), as in demo_01

out_dir <- file.path(get_script_dir(), Sys.getenv("RDUMP_DIR", "rdump"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------------------------
# DGP -- byte-for-byte identical to demo_trex_spca_01.R
# ------------------------------------------------------------------------------
dgp_sparse_factor_model <- function(n = 50, p = 100, p1 = 5, M = 3,
                                    target_snr_db = 0, overlap_pool = 30) {
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

  S <- Z %*% t(V)
  signal_var <- var(as.vector(S))

  E_raw <- matrix(rnorm(n * p), nrow = n, ncol = p)
  raw_noise_var <- var(as.vector(E_raw))

  target_noise_var <- signal_var / (10^(target_snr_db / 10))
  E <- E_raw * sqrt(target_noise_var / raw_noise_var)

  X <- S + E
  X <- scale(X, center = TRUE, scale = FALSE)
  list(X = X, V = V, Z = Z)
}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
# Full-precision CSV dump (17 significant digits round-trips IEEE double exactly).
write_matrix_full <- function(X, path) {
  Xc <- format(X, digits = 17, trim = TRUE, scientific = TRUE)
  write.table(Xc, file = path, sep = ",",
              row.names = FALSE, col.names = FALSE, quote = FALSE)
}

dash <- function(idx0) paste(idx0, collapse = "-")

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------
cat("================================================================\n")
cat("  R reference dump for C++ head-to-head  (T-Rex EN, PC1)\n")
cat("================================================================\n")
cat(sprintf("  out_dir : %s\n", out_dir))
cat(sprintf("  n=%d p=%d p1=%d M=%d SNR=%ddB tFDR=%.2f num_MC=%d\n\n",
            n, p, p1, M, snr_db, target_fdr, num_MC))

truth_rows     <- character(num_MC)
lam2_rows      <- character(num_MC)
res_rows       <- character(num_MC)   # type = "lar"   (default; what R's trex uses)
res_rows_lasso <- character(num_MC)   # type = "lasso" (LARS-LASSO; can drop variables)
pc1_rows       <- character(num_MC)

sum_k <- 0; sum_fdr <- 0; sum_tpr <- 0; sum_lam2 <- 0
sum_k_l <- 0; sum_fdr_l <- 0; sum_tpr_l <- 0   # lasso accumulators

for (mc in 0:(num_MC - 1)) {
  set.seed(seed_base + mc * 1000 + snr_db)
  d      <- dgp_sparse_factor_model(n, p, p1, M, snr_db, overlap_pool)
  X      <- d$X
  V_true <- d$V

  # ---- dump the design (full precision, headerless) ----
  write_matrix_full(X, file.path(out_dir, sprintf("X_%d.csv", mc)))

  # ---- PC1 truth (0-based) ----
  true_supp1 <- which(V_true[, 1] != 0)          # 1-based
  truth_rows[mc + 1] <- paste0(mc, ",", dash(true_supp1 - 1L))

  # ---- ordinary PCA -> PC1 score (== demo_01) ----
  svd_X <- svd(X)
  V_ord <- svd_X$v[, 1:M]
  Z_ord <- X %*% V_ord
  y1    <- Z_ord[, 1]
  pc1_rows[mc + 1] <- paste0(mc, ",",
                             paste(format(y1, digits = 17, trim = TRUE,
                                          scientific = TRUE), collapse = ","))

  # ---- lambda_2_lars via cv.glmnet (== demo_01 recipe) ----
  cvfit <- glmnet::cv.glmnet(x = X, y = y1, intercept = FALSE,
                             standardize = TRUE, alpha = 0,
                             type.measure = "mse", family = "gaussian",
                             nfolds = 10)
  lam2 <- cvfit$lambda.1se * ncol(X) / 2          # n*(1-alpha)/2, alpha=0, n=ncol(X)
  lam2_rows[mc + 1] <- paste0(mc, ",", format(lam2, digits = 17,
                                              trim = TRUE, scientific = TRUE))

  # ---- T-Rex+GVS (Elastic Net) selection on PC1 (== demo_01) ----
  # Run BOTH inner solvers on the SAME X, y1, lambda2 so the ONLY difference is
  # the inner forward-selection algorithm. TRexGVSSelector derives its solver
  # from gvs_type = "EN"; the legacy type switch therefore maps to en_solver:
  #   type = "lar"   -> en_solver = "TENET"     (pure forward; never drops a var)
  #   type = "lasso" -> en_solver = "TENET_AUG" (LARS-LASSO / augmented, can drop)
  # Selector seed is deterministic (mc + 1) so the dump reproduces across runs.
  trex_res <- TRexGVSSelector$new(
    X = X, y = y1, tFDR = target_fdr, seed = as.integer(mc + 1L), verbose = FALSE,
    gvs_control = trex_gvs_control(gvs_type = "EN", corr_max = 0.5,
                                   hc_linkage = "Single", lambda_2 = lam2,
                                   en_solver = "TENET"),
    control = trex_control(solver = "TLARS"))
  trex_res$select()

  active_set <- trex_res$selected_indices   # 1-based
  k  <- length(active_set)
  tp <- length(intersect(true_supp1, active_set))
  tpr <- if (length(true_supp1) == 0) 0 else tp / length(true_supp1)
  fdr <- if (k == 0) 0 else (k - tp) / k

  # num_dummies / rho_thresh are not surfaced by the R6 binding -> NA (rho_thresh
  # was already NA under the CRAN package); T_stop and v_thresh are preserved.
  vthr <- if (is.null(trex_res$v_thresh)) NA else trex_res$v_thresh[1]
  rthr <- NA
  res_rows[mc + 1] <- paste0(mc, ",", k, ",",
                             format(fdr, digits = 6), ",",
                             format(tpr, digits = 6), ",",
                             trex_res$T_stop, ",", NA, ",",
                             format(vthr, digits = 8), ",",
                             format(rthr, digits = 8), ",",
                             dash(active_set - 1L))

  sum_k <- sum_k + k; sum_fdr <- sum_fdr + fdr
  sum_tpr <- sum_tpr + tpr; sum_lam2 <- sum_lam2 + lam2

  # ---- same selection, but with en_solver = "TENET_AUG" (LARS-LASSO) ----
  trex_res_l <- TRexGVSSelector$new(
    X = X, y = y1, tFDR = target_fdr, seed = as.integer(mc + 1L), verbose = FALSE,
    gvs_control = trex_gvs_control(gvs_type = "EN", corr_max = 0.5,
                                   hc_linkage = "Single", lambda_2 = lam2,
                                   en_solver = "TENET_AUG"),
    control = trex_control(solver = "TLARS"))
  trex_res_l$select()

  active_set_l <- trex_res_l$selected_indices  # 1-based
  k_l  <- length(active_set_l)
  tp_l <- length(intersect(true_supp1, active_set_l))
  tpr_l <- if (length(true_supp1) == 0) 0 else tp_l / length(true_supp1)
  fdr_l <- if (k_l == 0) 0 else (k_l - tp_l) / k_l

  vthr_l <- if (is.null(trex_res_l$v_thresh)) NA else trex_res_l$v_thresh[1]
  rthr_l <- NA
  res_rows_lasso[mc + 1] <- paste0(mc, ",", k_l, ",",
                                   format(fdr_l, digits = 6), ",",
                                   format(tpr_l, digits = 6), ",",
                                   trex_res_l$T_stop, ",", NA, ",",
                                   format(vthr_l, digits = 8), ",",
                                   format(rthr_l, digits = 8), ",",
                                   dash(active_set_l - 1L))

  sum_k_l <- sum_k_l + k_l; sum_fdr_l <- sum_fdr_l + fdr_l
  sum_tpr_l <- sum_tpr_l + tpr_l

  if ((mc + 1) %% 10 == 0) cat(sprintf("  ... %d / %d trials\n", mc + 1, num_MC))
}

# ------------------------------------------------------------------------------
# Write dumps
# ------------------------------------------------------------------------------
writeLines(c("mc,indices",           truth_rows), file.path(out_dir, "truth.csv"))
writeLines(c("mc,lambda2",           lam2_rows),  file.path(out_dir, "r_lambda2.csv"))
writeLines(c("mc,k,fdr,tpr,T_stop,num_dummies,v_thresh,rho_thresh,indices", res_rows),
           file.path(out_dir, "r_results.csv"))
writeLines(c("mc,k,fdr,tpr,T_stop,num_dummies,v_thresh,rho_thresh,indices", res_rows_lasso),
           file.path(out_dir, "r_results_lasso.csv"))
writeLines(pc1_rows,                               file.path(out_dir, "r_pc1.csv"))

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
cat("\n----------------------------------------------------------------\n")
cat("  type = 'lar'  (LARS, R/trex default):\n")
cat(sprintf("    R mean k    : %.3f\n", sum_k    / num_MC))
cat(sprintf("    R mean FDR  : %.4f\n", sum_fdr  / num_MC))
cat(sprintf("    R mean TPR  : %.4f\n", sum_tpr  / num_MC))
cat("  type = 'lasso' (LARS-LASSO, can drop variables):\n")
cat(sprintf("    R mean k    : %.3f\n", sum_k_l   / num_MC))
cat(sprintf("    R mean FDR  : %.4f\n", sum_fdr_l / num_MC))
cat(sprintf("    R mean TPR  : %.4f\n", sum_tpr_l / num_MC))
cat(sprintf("  R mean lambda2: %.1f\n", sum_lam2 / num_MC))
cat(sprintf("\n  wrote: X_0..%d.csv, truth.csv, r_lambda2.csv, r_results.csv, r_results_lasso.csv, r_pc1.csv\n",
            num_MC - 1))
cat(sprintf("         into %s\n", out_dir))
cat("\n  NEXT (C++ head-to-head on identical X + identical lambda2):\n")
cat("    validation_trex_spca_04_rdump_pipeline --use-r-lambda2 --n 100\n")
cat("  then diff cpp_pipeline.csv vs r_results.csv (mean k, FDR, selections).\n")
