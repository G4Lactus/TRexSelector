# =======================================================================================
# Unit tests for the R6 class TRexSelector, focusing on active bindings and getters.
# =======================================================================================

test_that("TRexSelector active bindings enforce read-only protection", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexSelector$new(X, y, verbose = FALSE)

  # Try to assign to an active binding (which should fail)
  expect_error(
    selector$v_thresh <- 0.5,
    "v_thresh is read-only"
  )
  expect_error(
    selector$selected_var <- c(TRUE, FALSE),
    "selected_var is read-only"
  )
  expect_error(
    selector$fdp_hat_mat <- matrix(0),
    "fdp_hat_mat is read-only"
  )
  expect_error(
    selector$.ptr <- NULL,
    ".ptr is read-only"
  )
})


test_that("TRexSelector active bindings return expected data types post-selection", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexSelector$new(X, y, verbose = FALSE, control = trex_control(K = 3))

  # Before execution, properties may understandably be empty or matrix of 0
  # We invoke the algorithm directly
  selector$select()

  # Check vector state returns
  expect_type(selector$selected_indices, "integer")
  expect_type(selector$selected_var, "logical")
  expect_length(selector$selected_var, p)

  # Check numeric values
  expect_type(selector$v_thresh, "double")
  expect_type(selector$T_stop, "integer")
  expect_type(selector$L, "integer")
  expect_type(selector$y_mean, "double")

  # Check vector distributions
  expect_type(selector$x_means, "double")
  expect_length(selector$x_means, p)

  # Check the large matrix returns
  expect_true(is.matrix(selector$fdp_hat_mat))
  expect_true(is.matrix(selector$phi_mat))

})


test_that("TRexSelector selected_indices are 1-based and in [1, p]", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexSelector$new(X, y, verbose = FALSE, control = trex_control(K = 3))
  selector$select()

  idx <- selector$selected_indices
  expect_type(idx, "integer")
  expect_true(all(idx >= 1L & idx <= p))
})


test_that("TRexSelector selected_var and selected_indices are consistent", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexSelector$new(X, y, verbose = FALSE, control = trex_control(K = 3))
  selector$select()

  idx <- selector$selected_indices
  sv  <- selector$selected_var

  # selected_var has one entry per predictor
  expect_length(sv, p)
  # selected_indices == which(selected_var) (both 1-based)
  expect_equal(sort(idx), which(sv))
})


test_that("TRexSelector phi_prime is at most 1 with length p", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexSelector$new(X, y, verbose = FALSE, control = trex_control(K = 3))
  selector$select()

  phi_p <- selector$phi_prime
  expect_length(phi_p, p)
  expect_true(all(phi_p <= 1.0))  # capped at 1, may be negative
})


test_that("TRexSelector phi_mat has T_stop rows and p columns", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexSelector$new(X, y, verbose = FALSE, control = trex_control(K = 3))
  selector$select()

  ts <- selector$T_stop
  pm <- selector$phi_mat
  if (ts > 0L) {
    expect_equal(nrow(pm), ts)
    expect_equal(ncol(pm), p)
  }
})


test_that("TRexSelector multiple solver methods return 1-based indices consistent with selected_var", {
  n <- 100
  p <- 30
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 5] * 4.0 + X[, 15] * (-2.5) + rnorm(n)

  for (method in c("TLARS", "TLASSO", "TOMP")) {
    sel <- TRexSelector$new(X, y, tFDR = 0.1, verbose = FALSE,
                            control = trex_control(K = 20, solver = method))
    sel$select()

    idx <- sel$selected_indices
    sv  <- sel$selected_var

    expect_type(idx, "integer")
    expect_length(sv, p)

    expect_true(all(idx >= 1L & idx <= p), label = paste(method, "1-based in [1,p]"))
    expect_equal(sort(idx), which(sv), label = paste(method, "consistency"))
  }
})


test_that("exchangeable tie-breaking is plumbed through for greedy solvers", {
  # Control defaults: feature off, floor 0.5.
  ctrl <- trex_control()
  expect_equal(ctrl$exch_tie_alpha, 0)
  expect_equal(ctrl$exch_tie_floor, 0.5)

  # A correlated cluster so the exchangeable tie set is non-trivial.
  n <- 100
  p <- 12
  set.seed(42)
  base <- rnorm(n)
  X <- matrix(rnorm(n * p), n, p)
  for (j in 1:4) X[, j] <- base + 0.3 * rnorm(n)
  y <- base * 3.0 + rnorm(n)

  # Seeded selector (dummies and tie-breaks) and an unconditional
  # expectation, so the expectation count of the suite does not depend on
  # whether this run happened to select anything.
  for (method in c("TOMP", "TAFS")) {
    sel <- TRexSelector$new(
      X, y, tFDR = 0.2, seed = 1, verbose = FALSE,
      control = trex_control(K = 5, solver = method,
                             exch_tie_alpha = 0.25, exch_tie_floor = 0.5)
    )
    sel$select()
    idx <- sel$selected_indices
    expect_true(all(idx >= 1L & idx <= p),
                label = paste(method, "tie-broken run in [1,p]"))
  }
})


# =============================================================================
# restore(): give X back before select()
# =============================================================================

test_that("TRexSelector$restore() gives X back before select(); select() then errors", {
  set.seed(51)
  n <- 50; p <- 30
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 2 + rnorm(n)
  X0 <- X + 0
  sel <- TRexSelector$new(X, y, tFDR = 0.2, verbose = FALSE, control = trex_control(K = 5))
  expect_false(isTRUE(all.equal(X, X0)))          # normalized in place at construction
  sel$restore()
  expect_equal(X, X0, tolerance = 1e-12)
  sel$restore()                                    # idempotent
  expect_equal(X, X0, tolerance = 1e-12)
  expect_error(sel$select(), "restore")
  # A fresh selector on the same matrix runs and restores X itself
  sel2 <- TRexSelector$new(X, y, tFDR = 0.2, verbose = FALSE, control = trex_control(K = 5))
  sel2$select()
  expect_equal(X, X0, tolerance = 1e-12)
  sel2$restore()                                   # no-op after select()
  expect_equal(X, X0, tolerance = 1e-12)
})

test_that("the derived selectors inherit restore()", {
  set.seed(52)
  n <- 40; p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 2] * 2 + rnorm(n)
  X0 <- X + 0
  sel <- TRexScreeningSelector$new(X, y, seed = 3, verbose = FALSE)
  sel$restore()
  expect_equal(X, X0, tolerance = 1e-12)
  expect_error(sel$select(), "restore")
})

test_that("the selector control getters and num_dummies are exposed", {
  set.seed(53)
  n <- 40; p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 2 + rnorm(n)
  sel <- TRexSelector$new(X, y, tFDR = 0.2, verbose = FALSE,
                          control = trex_control(K = 5, max_dummy_multiplier = 2))
  expect_equal(sel$tFDR, 0.2)
  expect_identical(sel$K, 5L)
  expect_identical(sel$max_num_dummies, 2L)
  expect_type(sel$max_T_stop, "logical")
  expect_type(sel$stagnation_check, "logical")
  sel$select()
  expect_identical(sel$num_selected, length(sel$selected_indices))
  expect_true(sel$num_dummies >= p)
})

test_that("SKIPL with a large budget warns only without the stagnation stop", {
  set.seed(42)
  X <- matrix(rnorm(50 * 10), 50, 10)
  y <- rnorm(50)
  warns <- function(strategy, multiplier, stop = NULL) {
    ctrl <- trex_control(solver = "TLARS", K = 3, lloop_strategy = strategy,
                         max_dummy_multiplier = multiplier, tloop_stagnation_stop = stop)
    out <- utils::capture.output(
      TRexSelector$new(X, y, verbose = FALSE, control = ctrl)$select(),
      type = "message"
    )
    any(grepl("lloop_strategy SKIPL", out, fixed = TRUE))
  }
  expect_true(warns("SKIPL", 10))
  expect_false(warns("SKIPL", 5))
  expect_false(warns("SKIPL", 20, TRUE))
  expect_false(warns("STANDARD", 20))
})
