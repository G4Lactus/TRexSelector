# =============================================================================
# test-tsolvers-memory.R
# =============================================================================
#
# $get_memory_bytes(): the heap bytes a T-solver holds (the C++ memoryBytes()).
# X and D are borrowed and not counted; the value is a double.
# Mirrors cpp/tests/test_tsolvers/ad/test_tsolvers_memory.cpp.
# =============================================================================

# Deterministic data, n = 20, p = L = 10
n <- 20L
p <- 10L
d <- 10L
X <- outer(seq_len(n), seq_len(p), function(i, j) sin(i + 2 * j))
D <- outer(seq_len(n), seq_len(d), function(i, j) cos(i + 3 * j))
y <- cos(seq_len(n))

reader <- tsolver_options(normalize = FALSE, intercept = FALSE)


test_that("the getter exists on the solver classes and returns a positive double", {
  groups <- rep(seq_len(5L), each = 2L)
  solvers <- list(
    TLARS_Solver$new(X, D, y, options = reader),
    TOMP_Solver$new(X, D, y, options = reader),
    TAFS_Solver$new(X, D, y, options = tsolver_options(normalize = FALSE, intercept = FALSE,
                                                       rho_afs = 0.5)),
    TCDLASSO_Solver$new(X, D, y, options = reader),
    TENETAug_Solver$new(X, D, y, options = tsolver_options(normalize = FALSE, intercept = FALSE,
                                                           lambda2 = 0.5)),
    TIENET_Solver$new(X, D, y, groups = groups,
                      options = tsolver_options(normalize = FALSE, intercept = FALSE,
                                                lambda2 = 0.5))
  )
  for (s in solvers) {
    bytes <- s$get_memory_bytes()
    expect_type(bytes, "double")
    expect_length(bytes, 1L)
    expect_gt(bytes, 0)
  }
})

test_that("TLARS matches the construction formula (reader, n = 20, p = L = 10)", {
  s <- TLARS_Solver$new(X, D, y, options = reader)
  P <- p + d
  M <- 8 * min(P, n)               # step limit 8 min(max_actives, effective_n)
  # y copy and residual (n), correlations, inactive list and Kahan
  # compensation (P each), one path step (48 bytes), one RSS, R2 and DoF
  # entry, dummy record and lambda sequence reserved for M + 1 steps
  expected <- 16 * n + 24 * P + 48 + 24 + 16 * (M + 1)
  expect_identical(s$get_memory_bytes(), expected)
})

test_that("the report grows with the path and excludes the borrowed X and D", {
  set.seed(3)
  Xr <- matrix(rnorm(40 * p), 40, p)
  Dr <- matrix(rnorm(40 * d), 40, d)
  yr <- Xr[, 1] * 2 - Xr[, 2] + rnorm(40)
  s <- TLARS_Solver$new(Xr, Dr, yr, options = reader)
  before <- s$get_memory_bytes()
  s$execute_step(T_stop = 3)
  expect_gt(s$get_memory_bytes(), before)

  # doubling n at P = 20 <= n moves only the y copy and the residual
  small <- TLARS_Solver$new(Xr, Dr, yr, options = reader)$get_memory_bytes()
  X2 <- rbind(Xr, Xr); D2 <- rbind(Dr, Dr); y2 <- c(yr, yr)
  large <- TLARS_Solver$new(X2, D2, y2, options = reader)$get_memory_bytes()
  expect_identical(large - small, 2 * 8 * 40)
})

test_that("TENETAug reports its owned augmented copies through the base class", {
  s <- TENETAug_Solver$new(X, D, y, options = tsolver_options(normalize = FALSE,
                                                              intercept = FALSE,
                                                              lambda2 = 0.5))
  P <- p + d
  owned <- 8 * (n + P) * (P + 1)   # augmented X, D and y
  expect_gt(s$get_memory_bytes(), owned)
})
