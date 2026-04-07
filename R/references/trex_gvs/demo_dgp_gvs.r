# ==============================================================================
# demo_dgp_gvs.R
#
# Sanity checks for all four GVS DGPs.
# Run from the project root directory where dgp_gvs.R lives.
#
# Usage:
#   Rscript demo_dgp_gvs.R
#   # or interactively:
#   source("demo_dgp_gvs.R")
# ==============================================================================

source("dgp_gvs.R")

cat("================================================================\n")
cat(" T-Rex+GVS — DGP demo\n")
cat("================================================================\n\n")

# ── Shared parameters ─────────────────────────────────────────────────────────

n         <- 100L
p         <- 200L
M         <- 10L
s         <- 1L
rho_in    <- 0.7
snr       <- 5.0
sigma_eps <- 1.0

# ── DGP 1: Block equicorrelation, equal group sizes ───────────────────────────

cat("--- DGP 1: Block equicorrelation (equal sizes) ---\n")
dgp1 <- dgp_block_equi(n = n, p = p, M = M, s = s,
                        rho_in = rho_in, snr = snr,
                        sigma_eps = sigma_eps, seed = 1L)

stopifnot(
  identical(dim(dgp1$X),  c(n, p)),
  length(dgp1$y)          == n,
  length(dgp1$beta)       == p,
  sum(dgp1$beta != 0)     == M * s,
  dgp1$M                  == M,
  all(dgp1$group_sizes    == p %/% M),
  all(abs(colMeans(dgp1$X)) < 1e-10)           # column-centred
)

# Within-group sample correlation should exceed between-group
sub1 <- dgp1$groups_list[[1]]
sub2 <- dgp1$groups_list[[2]]
rho_within  <- mean(cor(dgp1$X[, sub1])[upper.tri(matrix(0, length(sub1), length(sub1)))])
rho_between <- mean(abs(cor(dgp1$X[, sub1], dgp1$X[, sub2])))
stopifnot(rho_within > rho_between)

cat(sprintf("  X: %d x %d\n",      nrow(dgp1$X), ncol(dgp1$X)))
cat(sprintf("  M = %d groups, b = %d per group\n", dgp1$M, dgp1$group_sizes[1]))
cat(sprintf("  Active vars: %d\n",  length(dgp1$active_set)))
cat(sprintf("  Population SNR: %.2f\n", dgp1$SNR))
cat(sprintf("  Mean within-group |cor|:  %.3f (expected ~ %.2f)\n",
            rho_within, rho_in))
cat(sprintf("  Mean between-group |cor|: %.3f (expected ~ 0)\n", rho_between))
cat("  PASSED\n\n")

# ── DGP 2: AR(1) within groups ────────────────────────────────────────────────

cat("--- DGP 2: AR(1) within-group correlation ---\n")
dgp2 <- dgp_block_ar1(n = n, p = p, M = M, s = s,
                       rho_in = rho_in, snr = snr,
                       sigma_eps = sigma_eps, seed = 2L)

stopifnot(
  identical(dim(dgp2$X), c(n, p)),
  sum(dgp2$beta != 0)   == M * s
)

# Adjacent within-group correlation > distal within-group correlation (AR(1) decay)
sub <- dgp2$groups_list[[1]]
C   <- cor(dgp2$X[, sub])
adj  <- mean(C[abs(row(C) - col(C)) == 1])
dist <- mean(C[abs(row(C) - col(C)) == floor(length(sub) / 2)])
stopifnot(adj > dist)

cat(sprintf("  Adjacent within-group |cor|: %.3f\n",  adj))
cat(sprintf("  Distal  within-group |cor|:  %.3f (expected < adjacent)\n", dist))
cat(sprintf("  Population SNR: %.2f\n", dgp2$SNR))
cat("  PASSED\n\n")

# ── DGP 3: Unequal group sizes ────────────────────────────────────────────────

cat("--- DGP 3: Geometrically unequal group sizes ---\n")
p3    <- 126L   # sum of 2^0 + ... + 2^5 = 63; doubled for integer safety
M3    <- 6L
dgp3  <- dgp_block_unequal(n = n, p = p3, M = M3, s = s,
                            rho_in = rho_in, ratio = 2,
                            snr = snr, sigma_eps = sigma_eps, seed = 3L)

stopifnot(
  identical(dim(dgp3$X),   c(n, p3)),
  sum(dgp3$group_sizes)    == p3,
  dgp3$M                   == M3,
  length(unique(dgp3$group_sizes)) > 1   # sizes are indeed unequal
)

cat(sprintf("  Group sizes: %s\n",
            paste(sort(dgp3$group_sizes), collapse = ", ")))
cat(sprintf("  Active vars: %d (s=1 per group)\n", length(dgp3$active_set)))
cat(sprintf("  Population SNR: %.2f\n", dgp3$SNR))
cat("  PASSED\n\n")

# ── DGP 4: Weak between-group correlation ─────────────────────────────────────

cat("--- DGP 4: Between-group correlation stress test ---\n")
rho_out <- 0.3
dgp4    <- dgp_between_corr(n = n, p = p, M = M, s = s,
                             rho_in = rho_in, rho_out = rho_out,
                             snr = snr, sigma_eps = sigma_eps, seed = 4L)

# Sample between-group correlation should be > 0
sub1 <- dgp4$groups_list[[1]]
sub2 <- dgp4$groups_list[[3]]   # non-adjacent group
rho_btwn_sample <- mean(abs(cor(dgp4$X[, sub1], dgp4$X[, sub2])))
stopifnot(rho_btwn_sample > 0)

cat(sprintf("  rho_in  = %.2f,  rho_out = %.2f\n", rho_in, rho_out))
cat(sprintf("  Sample between-group |cor|: %.3f (expected > 0)\n", rho_btwn_sample))
cat(sprintf("  Population SNR: %.2f\n", dgp4$SNR))
cat("  PASSED\n\n")

# ── Summary table ─────────────────────────────────────────────────────────────

cat("================================================================\n")
cat(sprintf(" %-28s %-10s %-8s %-6s\n", "DGP", "n x p", "M  s", "SNR"))
for (nm in c("dgp1", "dgp2", "dgp3", "dgp4")) {
  d <- get(nm)
  cat(sprintf(" %-28s %3d x %-5d M=%d, s=%d %.2f\n",
              nm,
              nrow(d$X), ncol(d$X),
              d$M, d$params$s, d$SNR))
}
cat("\nAll checks PASSED.\n")

heatmap(cov(dgp1$X), symm = TRUE, main = "Covariance matrix of DGP 1")
heatmap(cov(dgp2$X), symm = TRUE, main = "Covariance matrix of DGP 2")
heatmap(cov(dgp3$X), symm = TRUE, main = "Covariance matrix of DGP 3")
heatmap(cov(dgp4$X), symm = TRUE, main = "Covariance matrix of DGP 4")
