#!/usr/bin/env Rscript
# Probe: is R's -10dB FDR=0.0975 reproducible, or a lucky dummy draw?
# Re-run T-Rex on the SAME rdump10 X files under several different selector
# seeds (which change the internal dummy realizations) and report the spread of
# mean FDR. Stable mean -> systematic; wide spread -> MC dummy noise.
#
# MIGRATED: exercises the TRexSelectorNeo R binding (TRexGVSSelector) rather
# than the CRAN TRexSelector package. The probed dummy-RNG seed is now passed
# directly as the selector's `seed` argument (was a global set.seed()).

suppressMessages(library(TRexSelectorNeo))

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args)
  if (length(m)) return(dirname(normalizePath(sub("^--file=", "", args[m]))))
  getwd()
}

dir   <- file.path(get_script_dir(), "rdump10")
n_mc  <- 100L
tFDR  <- 0.10

# --- exact dumped inputs ---
pc1      <- as.matrix(read.csv(file.path(dir, "r_pc1.csv"), header = FALSE)) # mc, n scores
lam2_df  <- read.csv(file.path(dir, "r_lambda2.csv"))                        # mc,lambda2
truth_df <- read.csv(file.path(dir, "truth.csv"), colClasses = "character")  # mc,indices(0-based)

truth_list <- lapply(seq_len(n_mc), function(mc) {
  s <- truth_df$indices[truth_df$mc == as.character(mc - 1)]
  as.integer(strsplit(s, "-")[[1]]) + 1L      # 0-based -> 1-based
})

fdp_tpp <- function(sel_idx, truth) {
  tp  <- length(intersect(sel_idx, truth))
  fp  <- length(setdiff(sel_idx, truth))
  fdp <- if (length(sel_idx) == 0) 0 else fp / length(sel_idx)
  tpp <- tp / length(truth)
  c(fdp = fdp, tpp = tpp, k = length(sel_idx))
}

seeds <- c(1, 7, 42, 123, 999, 2024)
cat(sprintf("Probing R dummy-RNG variance on identical -10dB X (%d trials each)\n", n_mc))
cat(sprintf("%-8s %-10s %-10s %-10s\n", "seed", "mean_FDR", "mean_TPR", "mean_k"))

for (s in seeds) {
  fdrs <- numeric(n_mc); tprs <- numeric(n_mc); ks <- numeric(n_mc)
  for (mc in seq_len(n_mc)) {
    X  <- as.matrix(read.csv(file.path(dir, sprintf("X_%d.csv", mc - 1)),
                             header = FALSE))
    y1 <- as.numeric(pc1[mc, -1])            # drop the mc index column
    lam <- lam2_df$lambda2[lam2_df$mc == (mc - 1)]
    # vary the internal dummy realization per (seed, trial) via the selector seed
    res <- TRexGVSSelector$new(
      X = X, y = y1, tFDR = tFDR, seed = as.integer(s * 100000L + mc),
      verbose = FALSE,
      gvs_control = trex_gvs_control(gvs_type = "EN", lambda_2 = lam,
                                     en_solver = "TENET"),
      control = trex_control(solver = "TLARS"))
    res$select()
    sel_idx <- res$selected_indices
    m <- fdp_tpp(sel_idx, truth_list[[mc]])
    fdrs[mc] <- m["fdp"]; tprs[mc] <- m["tpp"]; ks[mc] <- m["k"]
  }
  cat(sprintf("%-8d %-10.4f %-10.4f %-10.3f\n", s, mean(fdrs), mean(tprs), mean(ks)))
}
