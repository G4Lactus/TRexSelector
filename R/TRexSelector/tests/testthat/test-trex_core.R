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
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(K = 1)),
    "K must be >= 2"
  )


  # Invalid selector solver enum matching
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(solver = "INVALID_METHOD")),
    "Unknown TRex selector solver:"
  )


  # Invalid LLoop strategy enum matching
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE,
                     control = trex_control(solver = "SCREEN", lloop_strategy = "INVALID")),
    "Unknown LLoopStrategy:"
  )

  # max_dummy_multiplier
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(max_dummy_multiplier = 0)),
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
    TRexSelector$new(X, y, verbose = FALSE,
                     control = trex_control(use_openmp = TRUE, max_outer_threads = 2))
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
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(use_memory_mapping = TRUE))
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
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(opt_threshold = -0.5)),
    "opt_threshold"
  )
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(opt_threshold = 1.5)),
    "opt_threshold"
  )

  # tloop_max_stagnant_steps cannot be negative
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(tloop_max_stagnant_steps = -1)),
    "tloop_max_stagnant_steps"
  )

  # OpenMP threads cannot be negative
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(max_outer_threads = -1)),
    "max_outer_threads"
  )
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(max_inner_threads = -1)),
    "max_inner_threads"
  )

  # lambda2 (ENET constraint) and rho_afs (AFS constraint) bounds
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(lambda2 = -0.1)),
    "lambda2"
  )
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(rho_afs = -0.5)),
    "rho_afs"
  )

  # ncgmp_variant valid types (assumed boolean / specific int)
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(ncgmp_variant = 5)),
    "ncgmp_variant"
  )

  # Explicitly unsupported solver variants
  expect_error(
    TRexSelector$new(X, y, verbose = FALSE, control = trex_control(solver = "UNKNOWN_SOLVER")),
    "Unknown TRex selector solver"
  )
})

# =======================================================================================
# trex_dummy_distribution() — structural tests
# =======================================================================================

test_that("trex_dummy_distribution returns correct list structure for Normal (default)", {
  d <- trex_dummy_distribution()
  expect_type(d, "list")
  expect_equal(d$type, "Normal")
  expect_null(d$df)
})

test_that("trex_dummy_distribution returns correct list structure for StudentT", {
  d <- trex_dummy_distribution("StudentT", df = 3)
  expect_equal(d$type, "StudentT")
  expect_equal(d$df, 3)
})

test_that("trex_dummy_distribution returns correct list structure for Uniform", {
  d <- trex_dummy_distribution("Uniform", a = 2.0)
  expect_equal(d$type, "Uniform")
  expect_equal(d$a, 2.0)
})

test_that("trex_dummy_distribution rejects unknown type", {
  expect_error(trex_dummy_distribution("GaussianMixture"), "Unknown distribution type")
})

test_that("trex_control includes dummy_distribution field", {
  ctrl <- trex_control()
  expect_true("dummy_distribution" %in% names(ctrl))
  expect_equal(ctrl$dummy_distribution$type, "Normal")
})

test_that("trex_control passes custom dummy_distribution through", {
  ctrl <- trex_control(dummy_distribution = trex_dummy_distribution("Rademacher"))
  expect_equal(ctrl$dummy_distribution$type, "Rademacher")
})

# =======================================================================================
# trex_dummy_distribution() — integration tests (TRex actually runs)
# =======================================================================================

test_that("TRexSelector runs with Rademacher dummy distribution", {
  skip_on_cran()
  set.seed(1)
  n <- 60; p <- 12
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] + X[, 2] + rnorm(n)
  sel <- TRexSelector$new(X, y, tFDR = 0.2, verbose = FALSE,
                          control = trex_control(
                            K = 5,
                            dummy_distribution = trex_dummy_distribution("Rademacher")
                          ))
  result <- sel$select()
  expect_type(result$selected_var, "logical")
  expect_length(result$selected_var, p)
})

test_that("TRexSelector runs with StudentT dummy distribution (df=3)", {
  skip_on_cran()
  set.seed(2)
  n <- 60; p <- 12
  X <- matrix(rnorm(n * p), n, p)
  y <- X[, 1] + X[, 2] + rnorm(n)
  sel <- TRexSelector$new(X, y, tFDR = 0.2, verbose = FALSE,
                          control = trex_control(
                            K = 5,
                            dummy_distribution = trex_dummy_distribution("StudentT", df = 3)
                          ))
  result <- sel$select()
  expect_type(result$selected_var, "logical")
  expect_length(result$selected_var, p)
})
