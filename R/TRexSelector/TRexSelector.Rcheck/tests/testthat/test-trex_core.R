# =======================================================================================
# TRexSelector core - Unit tests
# =======================================================================================

test_that("TRexSelector correctly guards against invalid backend parameters", {
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # tFDR bounds constraints ([0, 1])
  expect_error(
    TRexSelector$new(X, y, tFDR = -0.1, verbose = FALSE),
    "tFDR must be in"
  )
  expect_error(
    TRexSelector$new(X, y, tFDR = 1.5, verbose = FALSE),
    "tFDR must be in"
  )


  # K value checks
  expect_error(
    TRexSelector$new(X, y, K = 1, verbose = FALSE),
    "K must be >= 2"
  )


  # Invalid selector method enum matching
  expect_error(
    TRexSelector$new(X, y, method = "INVALID_METHOD", verbose = FALSE),
    "Unknown TRex selector method:"
  )


  # Invalid LLoop strategy enum matching
  expect_error(
    TRexSelector$new(X, y, method = "SCREEN", lloop_strategy = "INVALID", verbose = FALSE),
    "Unknown LLoopStrategy:"
  )

  # max_dummy_multiplier
  expect_error(
    TRexSelector$new(X, y, max_dummy_multiplier = 0, verbose = FALSE),
    "max_dummy_multiplier must be >= 1. Got: 0"
  )
})


test_that("TRexSelector handles concurrent execution properly (OpenMP)", {
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Ensure the object can be created and bound appropriately with concurrency settings
  selector_openmp <- expect_no_error(
    TRexSelector$new(X, y, use_openmp = TRUE, max_outer_threads = 2, verbose = FALSE)
  )

  # Run the algorithm targeting out threaded branches in C++ and falls back cleanly
  # if openMP is disabled.
  res_openmp <- expect_no_error(selector_openmp$select())

  expect_type(res_openmp, "environment")
  expect_true("selected_indices" %in% names(res_openmp))
})


test_that("TRexSelector supports memory mapping workflow correctly", {
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # Activate the memmap_manager_ branch deeply embedded in the backend utilities
  selector_mmap <- expect_no_error(
    TRexSelector$new(X, y, use_memory_mapping = TRUE, verbose = FALSE)
  )

  # Validate the full data stream workflow holds mapping pointers open effectively
  res_mmap <- expect_no_error(selector_mmap$select())

  expect_type(res_mmap, "environment")
  expect_true("selected_indices" %in% names(res_mmap))
})


test_that("TRexSelector correctly guards against edge-case hyperparameter inputs", {

  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  # opt_threshold bounds constraints ([0, 1])
  expect_error(
    TRexSelector$new(X, y, opt_threshold = -0.5, verbose = FALSE),
    "opt_threshold"
  )
  expect_error(
    TRexSelector$new(X, y, opt_threshold = 1.5, verbose = FALSE),
    "opt_threshold"
  )

  # tloop_max_stagnant_steps cannot be negative
  expect_error(
    TRexSelector$new(X, y, tloop_max_stagnant_steps = -1, verbose = FALSE),
    "tloop_max_stagnant_steps"
  )

  # OpenMP threads cannot be negative
  expect_error(
    TRexSelector$new(X, y, max_outer_threads = -1, verbose = FALSE),
    "max_outer_threads"
  )
  expect_error(
    TRexSelector$new(X, y, max_inner_threads = -1, verbose = FALSE),
    "max_inner_threads"
  )

  # lambda2 (ENET constraint) and rho_afs (AFS constraint) bounds
  expect_error(
    TRexSelector$new(X, y, lambda2 = -0.1, verbose = FALSE),
    "lambda2"
  )
  expect_error(
    TRexSelector$new(X, y, rho_afs = -0.5, verbose = FALSE),
    "rho_afs"
  )

  # ncgmp_variant valid types (assumed boolean / specific int)
  expect_error(
    TRexSelector$new(X, y, ncgmp_variant = 5, verbose = FALSE),
    "ncgmp_variant"
  )

  # Explicitly unsupported method variants
  expect_error(
    TRexSelector$new(X, y, method = "UNKNOWN_SOLVER", verbose = FALSE),
    "Unknown TRex selector method"
  )
})
