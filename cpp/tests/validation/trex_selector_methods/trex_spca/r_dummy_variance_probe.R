# r_dummy_variance_probe.R
# Quantify the T-Rex selector's OWN dummy-RNG variance on the identical dumped X.
# Reuses rdump/X_*.csv + r_lambda2.csv + truth.csv (no DGP rerun).
# For several internal seeds, recompute the mean FDR -> compare to the C++ band.
#
# MIGRATED: exercises the TRexSelectorNeo R binding (TRexGVSSelector) rather than
# the CRAN TRexSelector package. The probed internal-RNG seed is now passed
# directly as the selector's `seed` argument (was a global set.seed()).

suppressMessages(library(TRexSelectorNeo))

get_script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", a, value = TRUE)
  if (length(f)) return(dirname(normalizePath(sub("^--file=", "", f[1]))))
  if (!is.null(sys.frames()[[1]]$ofile)) return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  getwd()
}
rdump <- file.path(get_script_dir(), "rdump")

num_MC     <- 100
target_fdr <- 0.10
M          <- 3

# ---- load truth + lambda2 ----
tr  <- read.csv(file.path(rdump, "truth.csv"),     stringsAsFactors = FALSE)
lam <- read.csv(file.path(rdump, "r_lambda2.csv"), stringsAsFactors = FALSE)
truth <- lapply(tr$indices, function(s) as.integer(strsplit(s, "-")[[1]]) + 1L)  # ->1-based
names(truth) <- tr$mc
lam2 <- setNames(lam$lambda2, lam$mc)

run_once <- function(seed_off) {
  fdrs <- numeric(num_MC)
  for (mc in 0:(num_MC - 1)) {
    X  <- as.matrix(read.csv(file.path(rdump, sprintf("X_%d.csv", mc)), header = FALSE))
    sv <- svd(X)
    y1 <- (X %*% sv$v[, 1:M])[, 1]
    # vary the selector's internal dummy RNG via its `seed` argument
    rr <- TRexGVSSelector$new(
      X = X, y = y1, tFDR = target_fdr,
      seed = as.integer(20240601 + seed_off * 100003 + mc), verbose = FALSE,
      gvs_control = trex_gvs_control(gvs_type = "EN", en_solver = "TENET",
                                     lambda_2 = lam2[[as.character(mc)]]),
      control = trex_control(solver = "TLARS"))
    rr$select()
    sel <- rr$selected_indices
    k   <- length(sel)
    tp  <- length(intersect(truth[[as.character(mc)]], sel))
    fdrs[mc + 1] <- if (k == 0) 0 else (k - tp) / k
  }
  mean(fdrs)
}

cat("R trex() dummy-RNG variance over 8 internal seeds (identical X + R lambda2):\n")
vals <- numeric(8)
for (s in 0:7) {
  vals[s + 1] <- run_once(s)
  cat(sprintf("  seed_off %d : mean FDR = %.4f\n", s, vals[s + 1]))
}
cat(sprintf("\n  R band: min=%.4f  max=%.4f  mean=%.4f\n", min(vals), max(vals), mean(vals)))
cat("  C++ band (seeds 1000..8000): 0.064 .. 0.076 (mean ~0.071)\n")
