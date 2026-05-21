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
  if (length(idx) > 0) {
    expect_true(all(idx >= 1L & idx <= p))
  }
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


test_that("TRexSelector phi_prime is in [0, 1] with length p", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexSelector$new(X, y, verbose = FALSE, control = trex_control(K = 3))
  selector$select()

  phi_p <- selector$phi_prime
  expect_length(phi_p, p)
  expect_true(all(phi_p >= 0.0 & phi_p <= 1.0))
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
                            control = trex_control(K = 20, method = method))
    sel$select()

    idx <- sel$selected_indices
    sv  <- sel$selected_var

    expect_type(idx, "integer")
    expect_length(sv, p)

    if (length(idx) > 0) {
      expect_true(all(idx >= 1L & idx <= p), label = paste(method, "1-based in [1,p]"))
    }
    expect_equal(sort(idx), which(sv), label = paste(method, "consistency"))
  }
})
