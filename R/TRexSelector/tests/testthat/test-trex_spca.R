# =============================================================================
# Unit tests for TRexSPCASelector (trex_spca_selector.R + rcpp_trex_spca.cpp)
#
# Test groups:
#   1. trex_spca_control — default and custom construction
#   2. TRexSPCASelector  — basic execution (ActiveSet / Thresholded)
#   3. TRexSPCASelector  — output dimensions
#   4. TRexSPCASelector  — active set bounds
#   5. TRexSPCASelector  — explained variance properties
#   6. TRexSPCASelector  — data integrity (X is restored after select())
#   7. TRexSPCASelector  — input validation
# =============================================================================

library(TRexSelector)

# Shared fixture: small reproducible dataset (kept small for speed)
local_fixture <- function() {
  set.seed(42)
  n <- 30L; p <- 10L
  list(
    X = matrix(rnorm(n * p), n, p),
    n = n, p = p
  )
}

M     <- 2L
tFDR  <- 0.2
ctrl  <- trex_spca_control(K = 3L, max_dummy_multiplier = 3L, seed = 1L)


# =============================================================================
# Group 1: trex_spca_control — constructor
# =============================================================================

test_that("trex_spca_control returns a named list with correct defaults", {
  ctrl_def <- trex_spca_control()
  expect_named(ctrl_def, c("mode", "lambda2", "seed", "K", "max_dummy_multiplier"))
  expect_equal(ctrl_def$mode,                 "ActiveSet")
  expect_equal(ctrl_def$lambda2,              1e-6)
  expect_equal(ctrl_def$seed,                 -1L)
  expect_equal(ctrl_def$K,                    20L)
  expect_equal(ctrl_def$max_dummy_multiplier, 10L)
})

test_that("trex_spca_control rejects invalid mode", {
  expect_error(trex_spca_control(mode = "unknown"))
})

test_that("trex_spca_control rejects negative lambda2", {
  expect_error(trex_spca_control(lambda2 = -1))
})


# =============================================================================
# Group 2: TRexSPCASelector — basic execution (modes)
# =============================================================================

test_that("TRexSPCASelector select() runs to completion (ActiveSet mode)", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = trex_spca_control(
    mode = "ActiveSet", K = 3L, max_dummy_multiplier = 3L, seed = 1L))
  expect_no_error(sel$select(M, tFDR))
})

test_that("TRexSPCASelector select() runs to completion (Thresholded mode)", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = trex_spca_control(
    mode = "Thresholded", K = 3L, max_dummy_multiplier = 3L, seed = 1L))
  expect_no_error(sel$select(M, tFDR))
})


# =============================================================================
# Group 3: TRexSPCASelector — output dimensions
# =============================================================================

test_that("TRexSPCASelector: score matrix Z has correct dimensions", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(M, tFDR)
  expect_equal(dim(sel$Z), c(d$n, M))
})

test_that("TRexSPCASelector: loading matrix V has correct dimensions", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(M, tFDR)
  expect_equal(dim(sel$V), c(d$p, M))
})

test_that("TRexSPCASelector: adjusted_ev and cumulative_ev have length M", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(M, tFDR)
  expect_equal(length(sel$adjusted_ev),  M)
  expect_equal(length(sel$cumulative_ev), M)
})

test_that("TRexSPCASelector: active_sets is a list of length M", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(M, tFDR)
  expect_type(sel$active_sets, "list")
  expect_equal(length(sel$active_sets), M)
})

test_that("TRexSPCASelector: M = 1 returns correct output shapes", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(1L, tFDR)
  expect_equal(dim(sel$Z), c(d$n, 1L))
  expect_equal(dim(sel$V), c(d$p, 1L))
  expect_equal(length(sel$active_sets), 1L)
  expect_equal(length(sel$adjusted_ev), 1L)
})


# =============================================================================
# Group 4: TRexSPCASelector — active set bounds
# =============================================================================

test_that("TRexSPCASelector: active set indices are within [1, p]", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(M, tFDR)
  for (as_m in sel$active_sets) {
    # An empty active set is a valid FDR-conservative outcome.
    expect_true(length(as_m) >= 0)
    if (length(as_m) > 0) {
      expect_true(all(as_m >= 1L))
      expect_true(all(as_m <= d$p))
    }
  }
})

test_that("TRexSPCASelector: active set indices are integers", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(M, tFDR)
  for (as_m in sel$active_sets) {
    expect_type(as_m, "integer")
  }
})


# =============================================================================
# Group 5: TRexSPCASelector — explained variance properties
# =============================================================================

test_that("TRexSPCASelector: adjusted_ev is non-negative", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(M, tFDR)
  expect_true(all(sel$adjusted_ev >= 0))
})

test_that("TRexSPCASelector: cumulative_ev is non-decreasing and in [0, 1]", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(M, tFDR)
  ev <- sel$cumulative_ev
  expect_true(all(diff(ev) >= -1e-10))
  expect_true(all(ev >= 0))
  expect_true(all(ev <= 1 + 1e-10))
})


# =============================================================================
# Group 6: TRexSPCASelector — data integrity
# =============================================================================

test_that("TRexSPCASelector: X is restored after select()", {
  d <- local_fixture()
  # `d$X * 1.0` forces a deep copy; R's COW does not protect against direct
  # Rcpp/Eigen writes to the underlying SEXP memory.
  X_orig <- d$X * 1.0
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  sel$select(M, tFDR)
  # The C++ layer centers X in-place and restores it on return
  expect_equal(d$X, X_orig, tolerance = 1e-12)
})


# =============================================================================
# Group 7: TRexSPCASelector — input validation
# =============================================================================

test_that("TRexSPCASelector select() rejects M < 1", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  expect_error(sel$select(0L, tFDR))
})

test_that("TRexSPCASelector select() rejects tFDR out of (0, 1)", {
  d <- local_fixture()
  sel <- TRexSPCASelector$new(d$X, control = ctrl)
  expect_error(sel$select(M, tFDR = 0))
  expect_error(sel$select(M, tFDR = 1))
})
