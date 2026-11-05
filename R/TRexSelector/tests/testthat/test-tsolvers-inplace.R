# =============================================================================
# test-tsolvers-inplace.R
# =============================================================================
#
# The in-place writer/reader contract of the R6 T-solvers.
#
# The solvers map the R matrices (and memory-mapped files) without copying.
#   - A WRITER (normalize or intercept TRUE) centers / scales X and D in place,
#     visible on the caller's matrix, and holds a claim on both until
#     $restore() or garbage collection; a second writer on a claimed matrix is
#     an error.
#   - A READER (both FALSE) never writes and never claims.
#   - $restore() writes X and D back (idempotent), releases the claim, and the
#     fitted path stays available; the GC finalizer restores a writer that was
#     never restored.
#   - y is never written.
# Mirrors cpp/tests/test_tsolvers/ad/test_tsolvers_buffer_guard.cpp.
# =============================================================================

n <- 60L
p <- 8L
d <- 8L
set.seed(7)
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * d), n, d)
y <- X[, 1] * 3.0 + X[, 2] * (-2.0) + rnorm(n)

writer <- tsolver_options(normalize = TRUE, intercept = TRUE)
reader <- tsolver_options(normalize = FALSE, intercept = FALSE)


test_that("a writer preprocesses the caller's X and D in place and restore() undoes it", {
  X0 <- X + 0; D0 <- D + 0; y0 <- y + 0          # snapshots for the comparison
  s <- TLARS_Solver$new(X, D, y, options = writer)

  # In place: the caller's matrices are now centered and unit-norm
  expect_false(isTRUE(all.equal(X, X0)))
  expect_false(isTRUE(all.equal(D, D0)))
  expect_equal(colMeans(X), rep(0, p), tolerance = 1e-12)
  expect_equal(sqrt(colSums(X^2)), rep(1, p), tolerance = 1e-12)
  expect_identical(y, y0)                         # y is never written

  s$execute_step(T_stop = 4)
  nsteps <- s$get_num_steps()
  beta <- s$get_beta()

  s$restore()
  expect_equal(X, X0, tolerance = 1e-12)
  expect_equal(D, D0, tolerance = 1e-12)
  # The fitted path stays available after restore()
  expect_identical(s$get_num_steps(), nsteps)
  expect_identical(s$get_beta(), beta)

  # Idempotent: a second restore() leaves the data where it is
  X1 <- X + 0
  s$restore()
  expect_identical(X, X1)
})


test_that("a second writer on a claimed matrix is an error; readers are fine; restore() releases", {
  s1 <- TLARS_Solver$new(X, D, y, options = writer)
  on.exit(s1$restore(), add = TRUE)

  expect_error(TLARS_Solver$new(X, D, y, options = writer), "already held")
  expect_error(TOMP_Solver$new(X, D, y, options = writer), "already held")
  expect_error(TRexSelector$new(X, y, tFDR = 0.2, verbose = FALSE,
                                control = trex_control(K = 3)), "already held")

  # A reader coexists with the writer (and runs on the preprocessed data)
  r <- TSTAGEWISE_Solver$new(X, D, y, options = reader)
  expect_no_error(r$execute_step(T_stop = 3))

  # After restore() the matrix is free for the next writer
  s1$restore()
  s2 <- TLASSO_Solver$new(X, D, y, options = writer)
  expect_no_error(s2$execute_step(T_stop = 3))
  s2$restore()
})


test_that("garbage collection restores a writer that was never restored", {
  X0 <- X + 0
  local({
    s <- TLARS_Solver$new(X, D, y, options = writer)
    s$execute_step(T_stop = 3)
    expect_false(isTRUE(all.equal(X, X0)))
  })
  gc()                                            # runs the finalizer
  expect_equal(X, X0, tolerance = 1e-12)
  # ... and released the claim
  s <- TLARS_Solver$new(X, D, y, options = writer)
  s$restore()
})


test_that("a writer that resumed from a checkpoint restores with the archived statistics", {
  ck <- tempfile(fileext = ".bin")
  on.exit(unlink(ck), add = TRUE)
  X0 <- X + 0; D0 <- D + 0

  s1 <- TSTAGEWISE_Solver$new(X, D, y, options = writer)
  s1$execute_step(T_stop = 3)
  s1$save(ck)
  s1$restore()

  # The same object type, constructed as a writer on the raw data, then loaded
  s2 <- TSTAGEWISE_Solver$new(X, D, y, options = writer)
  s2$load(ck)
  expect_identical(s2$get_num_steps(), s1$get_num_steps())
  s2$execute_step(T_stop = 6)
  s2$restore()
  expect_equal(X, X0, tolerance = 1e-12)
  expect_equal(D, D0, tolerance = 1e-12)
})


test_that("pre-scaled readers: reference, checkpoint and resumed paths are bit-identical", {
  ck <- tempfile(fileext = ".bin")
  on.exit(unlink(ck), add = TRUE)
  X0 <- X + 0; D0 <- D + 0

  sx <- MLLpNormScaler$new(); sx$fit_transform_inplace(X)
  sd <- MLLpNormScaler$new(); sd$fit_transform_inplace(D)
  on.exit({ sx$inverse_transform_inplace(X); sd$inverse_transform_inplace(D) }, add = TRUE)

  ref <- TSTAGEWISE_Solver$new(X, D, y, options = reader)
  ref$execute_step(T_stop = 10)

  part <- TSTAGEWISE_Solver$new(X, D, y, options = reader)
  part$execute_step(T_stop = 4)
  part$save(ck)

  res <- TSTAGEWISE_Solver$new(X, D, y, options = reader)
  res$load(ck)
  res$execute_step(T_stop = 10)

  expect_identical(ref$get_num_steps(), res$get_num_steps())
  expect_identical(ref$get_actions(), res$get_actions())
  expect_identical(ref$get_beta(), res$get_beta())

  # Readers never claimed: a writer may take the matrices right away
  w <- TLARS_Solver$new(X, D, y, options = writer)   # warns: already preprocessed
  w$restore()

  sx$inverse_transform_inplace(X); sd$inverse_transform_inplace(D)
  expect_equal(X, X0, tolerance = 1e-12)
  expect_equal(D, D0, tolerance = 1e-12)
})


test_that("a memory-mapped X is preprocessed in the file and restored there", {
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  X0 <- X + 0
  mm <- mem_matrix_from(X, f)
  expect_equal(as.matrix(mm), X0, tolerance = 0)

  s <- TLARS_Solver$new(mm, D, y, options = writer)
  Xf <- as.matrix(mm)
  expect_false(isTRUE(all.equal(Xf, X0)))       # the file was written in place
  expect_equal(colMeans(Xf), rep(0, p), tolerance = 1e-12)
  expect_identical(X, X0)                         # the R matrix is untouched

  s$execute_step(T_stop = 3)
  expect_gt(s$get_num_steps(), 0L)

  # The file is claimed like a matrix
  expect_error(TLARS_Solver$new(mm, D, y, options = writer), "already held")

  s$restore()
  expect_equal(as.matrix(mm), X0, tolerance = 1e-12)
})
