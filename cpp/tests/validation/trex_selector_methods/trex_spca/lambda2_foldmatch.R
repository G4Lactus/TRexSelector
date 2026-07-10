# ==============================================================================
# lambda2_foldmatch.R
# ==============================================================================
#
# DECISIVE head-to-head for the lambda.1se "deviation" between the C++
# coordinate-descent ridge CV (elastic_net_cv_gaussian, alpha = 0) and R's
# glmnet::cv.glmnet(alpha = 0).
#
# The earlier probe used DIFFERENT random folds on each side, so lambda.1se
# could only be compared up to fold noise (R's own lambda.1se has sd ~ 17 on
# this dataset). This script removes that confound: it fixes an explicit,
# deterministic `foldid` that the C++ side replicates exactly, runs cv.glmnet,
# and dumps the FULL CV curve (lambda grid, cvm, cvsd) plus lambda.min/1se.
#
# The companion validation_trex_spca_05_lambda2_foldmatch.cpp reads these dumps,
# re-runs the C++ library CV on the SAME (X, y), SAME lambda grid and SAME
# foldid, and compares cvm / cvsd point-by-point. If they match to numerical
# tolerance, CD == glmnet and the prior lambda.1se gap was pure fold noise.
# If they diverge, the divergence point localises the bug.
# ==============================================================================

suppressPackageStartupMessages(library(glmnet))

# ---- Locate the shared C++ dump directory ------------------------------------
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

n <- nrow(X); p <- ncol(X)
nfolds <- 10L

cat(sprintf("Loaded X: %d x %d,  y: %d   (nfolds=%d)\n", n, p, length(y), nfolds))

# ---- Deterministic foldid the C++ side replicates 1:1 ------------------------
# Assignment: observation i (0-based) -> fold (i %% nfolds).  Dumped 0-based so
# the C++ demo reads it verbatim; cv.glmnet wants 1-based, so we add 1 here.
foldid0 <- (seq_len(n) - 1L) %% nfolds          # 0..nfolds-1
foldid1 <- foldid0 + 1L                          # 1..nfolds  (for glmnet)

# ---- cv.glmnet on the fixed folds (alpha=0 ridge, SPCA preprocessing) --------
# standardize=TRUE, intercept=FALSE: X and y are already centered (PC1 score),
# matching the production SPCA path and the C++ library call (intercept=false).
# thresh tightened (via glmnet.control, the non-deprecated route) so glmnet's CD
# converges to ~machine-precision ridge, making the curve comparison reflect the
# algorithm rather than convergence slack.
old_ctrl <- glmnet::glmnet.control()
glmnet::glmnet.control(thresh = 1e-14)
on.exit(do.call(glmnet::glmnet.control, old_ctrl), add = TRUE)

cvfit <- cv.glmnet(
  x = X, y = y,
  alpha        = 0,
  standardize  = TRUE,
  intercept    = FALSE,
  type.measure = "mse",
  family       = "gaussian",
  nfolds       = nfolds,
  foldid       = foldid1,
  grouped      = TRUE
)

K <- length(cvfit$lambda)
idx_min <- which(cvfit$lambda == cvfit$lambda.min)[1]
idx_1se <- which(cvfit$lambda == cvfit$lambda.1se)[1]

# ---- Dump full curve for the C++ comparison ----------------------------------
wcsv <- function(v, name) {
  write.table(format(v, digits = 17, scientific = TRUE),
              file = file.path(cpp_results_dir, name),
              row.names = FALSE, col.names = FALSE, quote = FALSE)
}
wcsv(cvfit$lambda, "fm_r_lambda.csv")
wcsv(cvfit$cvm,    "fm_r_cvm.csv")
wcsv(cvfit$cvsd,   "fm_r_cvsd.csv")
wcsv(foldid0,      "fm_foldid.csv")          # 0-based, for the C++ side

# scalar summary
summ <- data.frame(
  K        = K,
  idx_min0 = idx_min - 1L,                    # 0-based index for C++
  idx_1se0 = idx_1se - 1L,
  lambda_min = cvfit$lambda.min,
  lambda_1se = cvfit$lambda.1se,
  cvm_min  = cvfit$cvm[idx_min],
  cvsd_min = cvfit$cvsd[idx_min]
)
write.csv(summ, file.path(cpp_results_dir, "fm_r_summary.csv"), row.names = FALSE)

cat("==============================================================\n")
cat("  R cv.glmnet (alpha=0, std=TRUE, intercept=FALSE, FIXED folds)\n")
cat("==============================================================\n")
cat(sprintf("  grid size K          = %d\n", K))
cat(sprintf("  lambda.min           = %.6f  (idx0 %d)\n", cvfit$lambda.min, idx_min - 1L))
cat(sprintf("  lambda.1se           = %.6f  (idx0 %d)\n", cvfit$lambda.1se, idx_1se - 1L))
cat(sprintf("  cvm[min]             = %.8f\n", cvfit$cvm[idx_min]))
cat(sprintf("  cvsd[min]            = %.8f\n", cvfit$cvsd[idx_min]))
cat(sprintf("  threshold (cvm+cvsd) = %.8f\n", cvfit$cvm[idx_min] + cvfit$cvsd[idx_min]))
cat(sprintf("  lambda_2_lars(1se)   = %.5f   (= lambda.1se * p/2)\n",
            cvfit$lambda.1se * p / 2))
cat("--------------------------------------------------------------\n")
cat(sprintf("  Dumped: fm_r_lambda.csv, fm_r_cvm.csv, fm_r_cvsd.csv,\n"))
cat(sprintf("          fm_foldid.csv, fm_r_summary.csv -> %s\n", cpp_results_dir))
cat(sprintf("  Now run: validation_trex_spca_05_lambda2_foldmatch\n"))
cat("==============================================================\n")
