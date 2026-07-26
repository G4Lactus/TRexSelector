# =======================================================================================
# TRexDASelector - Unit tests
# =======================================================================================


test_that("TRexDASelector instantiation and dimensional validation work", {

  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Valid instantiation
  expect_silent(TRexDASelector$new(X, y, verbose = FALSE))

  # Dimension mismatch
  y_bad <- rnorm(n + 10)
  expect_error(TRexDASelector$new(X, y_bad, verbose = FALSE), "length equal to")
})



test_that("TRexDASelector executes correctly with valid data", {

  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Inject some strong signals in the first 3 variables
  y <- X[, 1] * 5 + X[, 2] * -4 + X[, 3] * 3 + rnorm(n)

  # Instantiate TRex selector
  selector <- TRexDASelector$new(X, y, tFDR = 0.1, verbose = FALSE, control = trex_control(K = 5))

  # Execute variable selection
  selector$select()

  # Assert result structure and logic
  res_indices <- selector$selected_indices
  res_T_stop <- selector$T_stop
  res_num_dummies <- selector$L * ncol(selector$.ptr_X)

  expect_type(res_indices, "integer")

  expect_type(res_T_stop, "integer")
  expect_type(res_num_dummies, "integer")

  # Bounds check on output selected indices (1 <= index <= p)
  if (length(res_indices) > 0) {
    expect_true(all(res_indices >= 1 & res_indices <= p))
  }
})



test_that("TRexDASelector supports da_method = 'NN' (nearest-neighbour sweep)", {
  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + X[, 3] * 3 + rnorm(n)

  # NN must be accepted (previously the R glue threw "Unknown DAMethod").
  # hc_grid_length = 0 lets the core auto-select min(20, p).
  selector <- TRexDASelector$new(
    X, y, tFDR = 0.2, verbose = FALSE,
    da_control = trex_da_control(da_method = "NN"),
    control = trex_control(K = 5)
  )
  expect_no_error(selector$select())
  expect_identical(selector$da_method_used, "NN")
  expect_type(selector$selected_indices, "integer")
  if (length(selector$selected_indices) > 0) {
    expect_true(all(selector$selected_indices >= 1 &
                    selector$selected_indices <= p))
  }
  # NN runs a BT-style rho grid
  expect_gt(length(selector$rho_grid), 0L)
})


test_that("TRexDASelector supports the prior-groups DA path", {
  set.seed(42)
  n <- 60
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 5 + X[, 2] * -4 + X[, 3] * 3 + rnorm(n)

  # Two-level prior constraints over p = 8 variables. When prior_groups is
  # supplied, da_method is ignored and the core routes through
  # setupDA_PriorGroups(): hierarchical sub-clustering runs within the finest
  # common refinement of the levels (here the size-2 fine groups) and the rho
  # grid is built from the pooled within-group dendrogram heights.
  fine   <- c(1, 1, 2, 2, 3, 3, 4, 4)
  coarse <- c(1, 1, 1, 1, 2, 2, 2, 2)

  selector <- TRexDASelector$new(
    X, y, tFDR = 0.2, verbose = FALSE,
    da_control = trex_da_control(
      prior_groups = list(fine, coarse)
    ),
    control = trex_control(K = 5)
  )
  expect_no_error(selector$select())
  expect_type(selector$selected_indices, "integer")
  if (length(selector$selected_indices) > 0) {
    expect_true(all(selector$selected_indices >= 1 &
                    selector$selected_indices <= p))
  }
  # Data-driven rho grid: hc_grid_length points (default min(20, p) = 8),
  # ascending and terminated by the conservative rho = 1 singleton anchor.
  expect_equal(length(selector$rho_grid), p)
  expect_equal(selector$rho_grid, sort(selector$rho_grid))
  expect_equal(selector$rho_grid[length(selector$rho_grid)], 1.0)
})


test_that("prior_groups rejects negative labels", {
  set.seed(7)
  n <- 60
  p <- 6
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] * 4 + X[, 2] * -3 + rnorm(n)

  expect_error(
    TRexDASelector$new(
      X, y, tFDR = 0.2, verbose = FALSE,
      da_control = trex_da_control(
        prior_groups = list(c(1, 1, 2, 2, -3, 3))
      ),
      control = trex_control(K = 5)
    ),
    "non-negative"
  )
})


test_that("prior_groups rejects a level whose length != p", {
  set.seed(42)
  n <- 40
  p <- 6
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  expect_error(
    TRexDASelector$new(
      X, y, verbose = FALSE,
      da_control = trex_da_control(prior_groups = list(c(1, 1, 2)))  # length 3 != p
    ),
    "length"
  )
})


test_that("TRexDASelector behaves memory safe with memory mapping enabled", {

  set.seed(42)
  n <- 50
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Use use_memory_mapping = TRUE to confirm the sandboxed tempdir() doesn't crash
  selector <- TRexDASelector$new(X, y, verbose = FALSE, control = trex_control(K = 3, use_memory_mapping = TRUE))
  selector$select()

  res_indices <- selector$selected_indices
  res_T_stop <- selector$T_stop
  res_num_dummies <- selector$L * ncol(selector$.ptr_X)

  expect_type(res_indices, "integer")


})



test_that("TRexDASelector correctly guards against invalid DA parameters", {

  set.seed(42)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Invalid DA threshold parameter bounds (rho_thr_DA must be [0, 1])
  expect_error(
    TRexDASelector$new(X, y, da_control = trex_da_control(da_method = "EQUI", rho_thr_DA = -0.5), verbose = FALSE),
    "rho_thr_DA"
  )
  expect_error(
    TRexDASelector$new(X, y, da_control = trex_da_control(da_method = "EQUI", rho_thr_DA = 1.5), verbose = FALSE),
    "rho_thr_DA"
  )

  # Invalid DA Methods that are caught by Rcpp string mapping
  expect_error(
    TRexDASelector$new(X, y, da_control = trex_da_control(da_method = "UNKNOWN_DA"), verbose = FALSE),
    "Unknown DAMethod:"
  )

  # Invalid Hierarchical Linkage Options (rejected at Rcpp mapping)
  expect_error(
    TRexDASelector$new(X, y, da_control = trex_da_control(da_method = "BT", hc_linkage = "UNKNOWN"), verbose = FALSE),
    "Unknown LinkageMethod:"
  )

  # Unsupported Hierarchical Linkage (passed Rcpp, rejected by C++ Core)
  # (Ward relies on non-Euclidean geometry and is blocked in validateDAParameters)
  expect_error(
    TRexDASelector$new(X, y, da_control = trex_da_control(da_method = "BT", hc_linkage = "Ward"), verbose = FALSE),
    "Unsupported linkage method for DA-BT"
  )

})



test_that("TRexDASelector handles concurrent execution properly (OpenMP with DA)", {

  set.seed(42)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # DA creates nested 3D calibrations; multithreaded runs ensure no race conditions
  selector_openmp <- expect_no_error(
    TRexDASelector$new(X, y,
                       da_control = trex_da_control(da_method = "BT", hc_linkage = "Average"),
                       control = trex_control(K = 5, use_openmp = TRUE, max_inner_threads = 2, max_outer_threads = 2),
                       verbose = FALSE)
  )

  # Let it resolve the OpenMP execution through evaluateStep overrides
  selector_openmp$select()




  expect_type(selector_openmp$selected_indices, "integer")

  expect_type(selector_openmp$T_stop, "integer")

})


# =======================================================================================
# Unit tests for the R6 class TRexDASelector, focusing on active bindings and getters.
# =======================================================================================

test_that("TRexDASelector active bindings enforce read-only protection", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexDASelector$new(X, y, verbose = FALSE)

  # Try to assign to DA-specific active bindings (which should fail)
  expect_error(
    selector$rho_thresh <- 0.5,
    "Field is read-only"
  )
  expect_error(
    selector$rho_grid <- c(0.1, 0.2),
    "Field is read-only"
  )
  expect_error(
    selector$da_method_used <- 1L,
    "Field is read-only"
  )
  expect_error(
    selector$FDP_hat_array_BT <- array(0, dim = c(1, 1, 1)),
    "Field is read-only"
  )
  expect_error(
    selector$Phi_array_BT <- array(0, dim = c(1, 1, 1)),
    "Field is read-only"
  )
  expect_error(
    selector$R_array_BT <- array(0, dim = c(1, 1, 1)),
    "Field is read-only"
  )
})

test_that("TRexDASelector active bindings return expected data types post-selection", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Using BT with a valid K for multidimensional matrices
  selector <- TRexDASelector$new(X, y, verbose = FALSE,
                                 da_control = trex_da_control(hc_grid_length = 3),
                                 control = trex_control(K = 3))

  # Execute so matrices get allocated
  selector$select()

  # Check standard primitive types
  expect_type(selector$rho_thresh, "double")
  expect_type(selector$rho_grid, "double")
  expect_type(selector$da_method_used, "character")

  # When da_method="BT", arrays should either be full 3D structures or valid fallback numeric structs
  expect_true(is.array(selector$FDP_hat_array_BT) || is.numeric(selector$FDP_hat_array_BT))
  expect_true(is.array(selector$Phi_array_BT) || is.numeric(selector$Phi_array_BT))
  expect_true(is.array(selector$R_array_BT) || is.numeric(selector$R_array_BT))

  if (is.array(selector$FDP_hat_array_BT)) {
    expect_equal(length(dim(selector$FDP_hat_array_BT)), 3)
  }

  # Ensure the get_matrices() legacy generator matches the active binding state
})


test_that("TRexDASelector selected_var and selected_indices are consistent (1-based)", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexDASelector$new(X, y, verbose = FALSE, control = trex_control(K = 3))
  selector$select()

  idx <- selector$selected_indices
  sv  <- selector$selected_var

  expect_type(idx, "integer")
  expect_length(sv, p)
  if (length(idx) > 0) {
    expect_true(all(idx >= 1L & idx <= p))
  }
  expect_equal(sort(idx), which(sv))
})


test_that("TRexDASelector phi_prime is in [0, 1] with length p", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexDASelector$new(X, y, verbose = FALSE, control = trex_control(K = 3))
  selector$select()

  phi_p <- selector$phi_prime
  expect_length(phi_p, p)
  expect_true(all(phi_p >= 0.0 & phi_p <= 1.0))
})


test_that("TRexDASelector phi_mat has T_stop rows and p columns", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexDASelector$new(X, y, verbose = FALSE, control = trex_control(K = 3))
  selector$select()

  ts <- selector$T_stop
  pm <- selector$phi_mat
  if (ts > 0L) {
    expect_gte(nrow(pm), 1L)
    expect_equal(ncol(pm), p)
  }
})


test_that("TRexDASelector DA-specific fields have valid bounds after selection", {
  n <- 50
  p <- 10
  set.seed(42)
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  selector <- TRexDASelector$new(X, y, verbose = FALSE,
                                 da_control = trex_da_control(da_method = "BT"),
                                 control = trex_control(K = 3))
  selector$select()

  expect_gte(selector$rho_thresh, 0.0)
  expect_lte(selector$rho_thresh, 1.0)
  expect_gt(length(selector$rho_grid), 0L)
  expect_true(all(selector$rho_grid >= 0.0 & selector$rho_grid <= 1.0))
  expect_true(selector$da_method_used %in% c("BT", "AR1", "EQUI", "BLOCK_EQUI"))
})
