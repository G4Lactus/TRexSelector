# ==============================================================================
# validation_ts_01_tenet_aug_comparison.R
# ==============================================================================
#
# Equivalence check between the Gram-based TENET_Solver, the augmented-LASSO
# TENETAug_Solver, and a manual elastic-net-to-LASSO augmentation solved with
# TLASSO_Solver (the R "lasso_star" recipe).
#
# Mirrors the co-located validation_ts_01_tenet_aug_comparison.cpp.
#
# Two scaling scenarios are checked on the same dataset:
#   Scenario 1: Z-score scaling (center + Bessel SD)
#   Scenario 2: L2-norm scaling (center + unit-L2 columns)
#
# All three solvers run their COMPLETE path (early_stop = FALSE); the beta
# paths are compared step by step in normalized space, and de-normalized back
# to the original scale for RMSE against the known true coefficients.
#
# Binding note: the R get_beta(step) accessor returns the X-block (first p
# coefficients) per step, which is exactly the block the cpp comparison uses;
# the path matrix is assembled by looping get_beta over steps 0..num_steps.
#
# ==============================================================================

library(TRexSelectorNeo)

this_dir_ <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    flag <- grep("^--file=", args, value = TRUE)
    if (length(flag) > 0L)
      dirname(normalizePath(sub("^--file=", "", flag[1L])))
    else "."
  }
)

source(file.path(this_dir_, "ts_demo_utils.R"))


# X-block beta path as a p x (num_steps + 1) matrix (column k = step k-1).
.beta_path_X <- function(solver) {
  ns <- solver$get_num_steps()
  vapply(0:ns, function(k) solver$get_beta(k),
         numeric(length(solver$get_beta(0))))
}


# ==============================================================================
# Comparison for one scaling scenario
# ==============================================================================

run_comparison <- function(scenario_name, X, D, y, true_beta_X, scaler_type,
                           lambda2, T_stop, p) {
  n <- nrow(X)
  L <- ncol(D)
  cat("\n", strrep("=", 78), "\n", scenario_name, "\n", strrep("=", 78), "\n",
      sep = "")

  # Per-column scale of the RAW X (before scaling), used to de-normalize the
  # coefficient paths back to the original scale:
  #   ZScore -> Bessel-corrected sample SD, L2 -> centered column L2 norm.
  scale_div <- if (scaler_type == "zscore") sqrt(n - 1) else 1.0
  Xc <- sweep(X, 2, colMeans(X))
  scale_X <- sqrt(colSums(Xc^2)) / scale_div

  # Apply the standardizer in place (separate instances for X and D).
  if (scaler_type == "zscore") {
    sc_X <- ZScoreScaler$new(center = TRUE, scale = TRUE)
    sc_D <- ZScoreScaler$new(center = TRUE, scale = TRUE)
  } else {
    sc_X <- LpNormScaler$new(norm_type = 2, center = TRUE)
    sc_D <- LpNormScaler$new(norm_type = 2, center = TRUE)
  }
  sc_X$fit(X)
  sc_X$transform_inplace(X)
  sc_D$fit(D)
  sc_D$transform_inplace(D)
  y <- y - mean(y)

  # TENET: Gram-based elastic net, full path.
  tenet <- TENET_Solver$new(X, D, y, lambda2, normalize = FALSE,
                            intercept = FALSE, verbose = FALSE)
  tenet$execute_step(T_stop, early_stop = FALSE)

  # TENETAug: augmented-LASSO elastic net, full path.
  tenet_aug <- TENETAug_Solver$new(X, D, y, lambda2, normalize = FALSE,
                                   intercept = FALSE, verbose = FALSE)
  tenet_aug$execute_step(T_stop, early_stop = FALSE)

  # Manual TLASSO on hand-augmented matrices (the R lasso_star recipe):
  #   d1 = sqrt(lambda2), d2 = 1/sqrt(1+lambda2)
  #   X_aug = [d2*X; d2*d1*I_p (rows n..n+p-1)],  D_aug analogous below it,
  #   y_aug = c(y, 0_{p+L}),  beta_original = beta_star / d2.
  d1 <- sqrt(lambda2)
  d2 <- 1 / sqrt(1 + lambda2)
  naug <- n + p + L
  X_aug <- matrix(0, naug, p)
  X_aug[1:n, ] <- d2 * X
  X_aug[(n + 1):(n + p), ] <- d2 * d1 * diag(p)
  D_aug <- matrix(0, naug, L)
  D_aug[1:n, ] <- d2 * D
  D_aug[(n + p + 1):naug, ] <- d2 * d1 * diag(L)
  y_aug <- c(y, rep(0, p + L))

  tlasso_aug <- TLASSO_Solver$new(X_aug, D_aug, y_aug, normalize = FALSE,
                                  intercept = FALSE, verbose = FALSE)
  tlasso_aug$execute_step(T_stop, early_stop = FALSE)

  # X-block beta paths: p x (steps+1); manual path de-normalized by 1/d2.
  path_tenet <- .beta_path_X(tenet)
  path_aug <- .beta_path_X(tenet_aug)
  path_manual <- .beta_path_X(tlasso_aug) / d2
  n_steps <- min(ncol(path_tenet), ncol(path_aug), ncol(path_manual))
  cat(sprintf("Path lengths: TENET = %d, TENET_AUG = %d, TLASSO_aug = %d\n",
              ncol(path_tenet), ncol(path_aug), ncol(path_manual)))

  inv_sqrt_p <- 1 / sqrt(p)
  rmse_ten <- rmse_aug <- rmse_man <- numeric(n_steps)
  d_ten_aug <- d_ten_man <- d_aug_man <- numeric(n_steps)
  for (k in seq_len(n_steps)) {
    bx_ten <- path_tenet[, k]
    bx_aug <- path_aug[, k]
    bx_man <- path_manual[, k]
    rmse_ten[k] <- sqrt(sum((bx_ten / scale_X - true_beta_X)^2)) * inv_sqrt_p
    rmse_aug[k] <- sqrt(sum((bx_aug / scale_X - true_beta_X)^2)) * inv_sqrt_p
    rmse_man[k] <- sqrt(sum((bx_man / scale_X - true_beta_X)^2)) * inv_sqrt_p
    d_ten_aug[k] <- sqrt(sum((bx_ten - bx_aug)^2))
    d_ten_man[k] <- sqrt(sum((bx_ten - bx_man)^2))
    d_aug_man[k] <- sqrt(sum((bx_aug - bx_man)^2))
  }

  # Sparse per-step table: every 5th step plus the last.
  cat(sprintf("%13s%13s%13s%13s%13s%13s%13s\n", "step", "RMSE_TEN", "RMSE_AUG",
              "RMSE_TLS", "d(TEN,AUG)", "d(TEN,TLS)", "d(AUG,TLS)"))
  for (k in seq_len(n_steps)) {
    step <- k - 1L
    if (step %% 5 != 0 && k != n_steps) next
    cat(sprintf("%13d%13.5f%13.5f%13.5f%13.2e%13.2e%13.2e\n",
                step, rmse_ten[k], rmse_aug[k], rmse_man[k],
                d_ten_aug[k], d_ten_man[k], d_aug_man[k]))
  }

  # Summary.
  cat("\nSummary:\n")
  cat(sprintf("  TENET     vs TENET_AUG : max L2 = %.4e, mean L2 = %.4e\n",
              max(d_ten_aug), mean(d_ten_aug)))
  cat(sprintf("  TENET     vs TLASSO_aug: max L2 = %.4e, mean L2 = %.4e\n",
              max(d_ten_man), mean(d_ten_man)))
  cat(sprintf("  TENET_AUG vs TLASSO_aug: max L2 = %.4e, mean L2 = %.4e\n",
              max(d_aug_man), mean(d_aug_man)))
  cat(sprintf("  TENET      RMSE: best = %.6f, final = %.6f\n",
              min(rmse_ten), rmse_ten[n_steps]))
  cat(sprintf("  TENET_AUG  RMSE: best = %.6f, final = %.6f\n",
              min(rmse_aug), rmse_aug[n_steps]))
  cat(sprintf("  TLASSO_aug RMSE: best = %.6f, final = %.6f\n",
              min(rmse_man), rmse_man[n_steps]))

  # Verdicts.
  verdict <- function(max_d) {
    if (max_d < 1e-8) return("EQUIVALENT")
    if (max_d < 1e-4) return("NEARLY EQUIVALENT")
    sprintf("DIFFER (max L2 = %.4e)", max_d)
  }
  cat("\nVerdicts:\n")
  cat(sprintf("  TENET     vs TENET_AUG : %s\n", verdict(max(d_ten_aug))))
  cat(sprintf("  TENET     vs TLASSO_aug: %s\n", verdict(max(d_ten_man))))
  cat(sprintf("  TENET_AUG vs TLASSO_aug: %s\n", verdict(max(d_aug_man))))

  max(d_ten_aug, d_ten_man, d_aug_man)
}


# ==============================================================================
# Main
# ==============================================================================

n <- 90L
p <- 150L
L <- 3L * p
lambda2 <- 100.01
T_stop <- 10L          # unused: early_stop = FALSE traces the complete path
snr <- 1.0

# cpp true_support is 0-based {10, 50, 85} -> 1-based here
true_support <- c(11L, 51L, 86L)
true_coefs <- c(2.5, -1.8, 3.2)
true_beta_X <- numeric(p)
true_beta_X[true_support] <- true_coefs

cat(sprintf("Generating data: n=%d p=%d L=%d lambda2=%g\n", n, p, L, lambda2))
dat <- gen_synthetic_data(n, p, L, true_support, true_coefs, snr, seed = 42)

worst <- 0.0
for (sc in list(list(name = "Scenario 1: Z-score scaling (centre + SD)",
                     type = "zscore"),
                list(name = "Scenario 2: L2-norm scaling (centre + L2-normalise)",
                     type = "l2"))) {
  # Fresh copies per scenario (scaling mutates in place).
  worst <- max(worst, run_comparison(sc$name, dat$X + 0, dat$D + 0, dat$y,
                                     true_beta_X, sc$type, lambda2, T_stop, p))
}

cat(sprintf("\nOverall worst pairwise max L2 = %.4e\n", worst))
