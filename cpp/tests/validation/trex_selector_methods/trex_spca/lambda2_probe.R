# ==============================================================================
# lambda2_probe.R
# ==============================================================================
#
# Cross-check for validation_trex_spca_01_lambda2_probe.cpp.
#
# Loads the EXACT centered X and PC1 score y that the C++ probe dumped, and
# computes glmnet's lambda.min / lambda.1se and the LARS-unit ridge penalty
#   lambda_2_lars = lambda.1se * ncol(X) / 2
# exactly as lm_dummy.R / demo_trex_spca_01.R do.
#
# Because glmnet's CV folds are random, lambda.1se has run-to-run variance on a
# single dataset. We therefore repeat the CV `n_rep` times with different seeds
# and report the mean +/- sd, so the comparison reflects the *scale* of
# lambda_2 rather than fold noise. Compare the printed mean against the
# `glmnet_lars` column of the C++ probe at the dumped SNR (default -10 dB).
# ==============================================================================

suppressPackageStartupMessages(library(glmnet))

# ---- Locate the C++ dump -----------------------------------------------------
# The C++ probe's DEMO_OUTPUT_DIR is the co-located validation_results/ folder.
# Resolve relative to this script so the probe works from any working directory.
this_dir_ <- tryCatch(
  dirname(normalizePath(sys.frame(1L)$ofile)),
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) dirname(normalizePath(sub("--file=", "", file_arg[1]))) else "."
  }
)
cpp_results_dir <- normalizePath(file.path(this_dir_, "validation_results"),
                                 mustWork = FALSE)

x_csv <- file.path(cpp_results_dir, "lambda2_probe_X.csv")
y_csv <- file.path(cpp_results_dir, "lambda2_probe_y.csv")

stopifnot(file.exists(x_csv), file.exists(y_csv))

X <- as.matrix(read.csv(x_csv, header = FALSE))
y <- scan(y_csv, quiet = TRUE)
storage.mode(X) <- "double"

cat(sprintf("Loaded X: %d x %d,  y: %d\n", nrow(X), ncol(X), length(y)))
cat(sprintf("Source: %s\n\n", cpp_results_dir))

# ---- Repeated cv.glmnet (alpha = 0), matching demo_trex_spca_01.R ------------
n_rep <- 25L
lam_min <- numeric(n_rep)
lam_1se <- numeric(n_rep)

for (r in seq_len(n_rep)) {
  set.seed(r)
  cvfit <- glmnet::cv.glmnet(x = X, y = y, intercept = FALSE,
                             standardize = TRUE, alpha = 0,
                             type.measure = "mse", family = "gaussian",
                             nfolds = 10)
  lam_min[r] <- cvfit$lambda.min
  lam_1se[r] <- cvfit$lambda.1se
}

p <- ncol(X)
lars_min <- lam_min * p / 2
lars_1se <- lam_1se * p / 2

fmt <- function(v) sprintf("%.5f (sd %.5f)", mean(v), sd(v))

cat("==============================================================\n")
cat("  R glmnet cv (alpha=0, standardize=TRUE, intercept=FALSE)\n")
cat(sprintf("  averaged over %d CV repetitions (random folds)\n", n_rep))
cat("==============================================================\n")
cat(sprintf("  lambda.min        = %s\n", fmt(lam_min)))
cat(sprintf("  lambda.1se        = %s\n", fmt(lam_1se)))
cat(sprintf("  lambda_2_lars(min)= %s\n", fmt(lars_min)))
cat(sprintf("  lambda_2_lars(1se)= %s   <-- compare to C++ glmnet_lars\n",
            fmt(lars_1se)))
cat("==============================================================\n")
