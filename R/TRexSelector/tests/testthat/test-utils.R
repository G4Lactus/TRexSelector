# =======================================================================================
# Utils - Unit tests for evaluation metric functions
# =======================================================================================


test_that("compute_fdp returns correct values for known inputs", {
  # 1 false discovery out of 3 selected: FDP = 1/3
  expect_equal(compute_fdp(c(1L, 2L, 3L), c(1L, 3L)), 1 / 3)

  # All selected are true positives: FDP = 0
  expect_equal(compute_fdp(c(1L, 3L), c(1L, 3L)), 0.0)

  # All selected are false discoveries: FDP = 1
  expect_equal(compute_fdp(c(2L, 4L), c(1L, 3L)), 1.0)
})


test_that("compute_fdp returns 0 for empty inputs", {
  expect_equal(compute_fdp(integer(0), c(1L, 2L)), 0.0)
  expect_equal(compute_fdp(c(1L, 2L), integer(0)), 0.0)
})


test_that("compute_tpp returns correct values for known inputs", {
  # Both true support found: TPP = 1
  expect_equal(compute_tpp(c(1L, 2L, 3L), c(1L, 3L)), 1.0)

  # Only 1 of 2 true positives found: TPP = 0.5
  expect_equal(compute_tpp(c(1L, 4L), c(1L, 3L)), 0.5)

  # None of the true support selected: TPP = 0
  expect_equal(compute_tpp(c(2L, 4L), c(1L, 3L)), 0.0)
})


test_that("compute_tpp returns 0 for empty inputs", {
  expect_equal(compute_tpp(integer(0), c(1L, 2L)), 0.0)
  expect_equal(compute_tpp(c(1L, 2L), integer(0)), 0.0)
})


test_that("compute_precision returns correct values for known inputs", {
  # 2 of 3 selected are true positives: precision = 2/3
  expect_equal(compute_precision(c(1L, 2L, 3L), c(1L, 3L)), 2 / 3)

  # All selected are true positives: precision = 1
  expect_equal(compute_precision(c(1L, 3L), c(1L, 3L)), 1.0)

  # No selected are true positives: precision = 0
  expect_equal(compute_precision(c(2L, 4L), c(1L, 3L)), 0.0)
})


test_that("compute_precision returns 0 for empty inputs", {
  expect_equal(compute_precision(integer(0), c(1L, 2L)), 0.0)
  expect_equal(compute_precision(c(1L, 2L), integer(0)), 0.0)
})


test_that("compute_recall returns correct values for known inputs", {
  # Both true support found: recall = 1
  expect_equal(compute_recall(c(1L, 2L, 3L), c(1L, 3L)), 1.0)

  # Only 1 of 2 found: recall = 0.5
  expect_equal(compute_recall(c(1L, 4L), c(1L, 3L)), 0.5)

  # None found: recall = 0
  expect_equal(compute_recall(c(2L, 4L), c(1L, 3L)), 0.0)
})


test_that("compute_recall returns 0 for empty inputs", {
  expect_equal(compute_recall(integer(0), c(1L, 2L)), 0.0)
  expect_equal(compute_recall(c(1L, 2L), integer(0)), 0.0)
})


test_that("compute_fdp_dense returns correct values for known inputs", {
  beta_hat <- c(1.0, 0.0, 1.0, 1.0)   # positions 1, 3, 4 non-zero
  beta     <- c(1.0, 0.0, 0.0, 1.0)   # true support: positions 1, 4

  # 1 false discovery (position 3) out of 3 selected: FDP = 1/3
  expect_equal(compute_fdp_dense(beta_hat, beta), 1 / 3)

  # All estimates match support: FDP = 0
  expect_equal(compute_fdp_dense(c(1.0, 0.0, 1.0), c(1.0, 0.0, 1.0)), 0.0)

  # All selected are false discoveries: FDP = 1
  expect_equal(compute_fdp_dense(c(0.0, 1.0, 0.0), c(1.0, 0.0, 1.0)), 1.0)
})


test_that("compute_fdp_dense returns 0 when nothing is selected", {
  expect_equal(compute_fdp_dense(c(0.0, 0.0), c(1.0, 0.0)), 0.0)
})


test_that("compute_tpp_dense returns correct values for known inputs", {
  beta_hat <- c(1.0, 0.0, 1.0, 1.0)
  beta     <- c(1.0, 0.0, 0.0, 1.0)

  # Both true positives (positions 1 and 4) recovered: TPP = 1
  expect_equal(compute_tpp_dense(beta_hat, beta), 1.0)

  # Only position 1 of 2 true positives found: TPP = 0.5
  expect_equal(compute_tpp_dense(c(1.0, 0.0, 0.0), c(1.0, 0.0, 1.0)), 0.5)

  # No true positive found: TPP = 0
  expect_equal(compute_tpp_dense(c(0.0, 1.0, 0.0), c(1.0, 0.0, 1.0)), 0.0)
})


test_that("metric functions return values in [0, 1]", {
  set.seed(1)
  selected <- sample(1:50, 10)
  support  <- sample(1:50, 10)

  expect_gte(compute_fdp(selected, support), 0.0)
  expect_lte(compute_fdp(selected, support), 1.0)

  expect_gte(compute_tpp(selected, support), 0.0)
  expect_lte(compute_tpp(selected, support), 1.0)

  expect_gte(compute_precision(selected, support), 0.0)
  expect_lte(compute_precision(selected, support), 1.0)

  expect_gte(compute_recall(selected, support), 0.0)
  expect_lte(compute_recall(selected, support), 1.0)
})


# =======================================================================================
# mmap_matrix - Element access (read) and write tests
# =======================================================================================

test_that("mmap_matrix single element read returns plain scalar", {
  mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  mm <- convert_to_memory_mapped(mat, f)

  # Must return a plain numeric, not a 1x1 matrix
  val <- mm[2, 3]
  expect_type(val, "double")
  expect_false(is.matrix(val))
  expect_equal(val, mat[2, 3])
})


test_that("mmap_matrix single element write round-trips correctly", {
  mat <- matrix(as.double(1:20), nrow = 4, ncol = 5)
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  mm <- convert_to_memory_mapped(mat, f)

  mm[2, 3] <- 99.0
  expect_equal(mm[2, 3], 99.0)
  # Neighbouring elements untouched
  expect_equal(mm[1, 3], mat[1, 3])
  expect_equal(mm[2, 4], mat[2, 4])
})


test_that("mmap_matrix block write round-trips correctly", {
  mat <- matrix(as.double(0), nrow = 4, ncol = 5)
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  mm <- convert_to_memory_mapped(mat, f)

  block <- matrix(as.double(1:6), nrow = 2, ncol = 3)
  mm[1:2, 2:4] <- block
  expect_equal(mm[1:2, 2:4], block)
  # Untouched region remains zero
  expect_equal(mm[3, 1], 0.0)
})


test_that("mmap_matrix write on readonly matrix raises error", {
  mat <- matrix(as.double(1:12), nrow = 3, ncol = 4)
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  convert_to_memory_mapped(mat, f)
  mm_ro <- mmap_matrix(f, rows = 3, cols = 4, mode = "readonly")
  expect_error(mm_ro[1, 1] <- 0.0, regexp = "read-only")
})


test_that("mmap_matrix element access respects bounds", {
  mat <- matrix(as.double(1:6), nrow = 2, ncol = 3)
  f <- tempfile(fileext = ".bin")
  on.exit(unlink(f), add = TRUE)
  mm <- convert_to_memory_mapped(mat, f)

  expect_error(mm[3, 1])  # row out of bounds
  expect_error(mm[1, 4])  # col out of bounds
  expect_error(mm[3, 1] <- 0.0)
  expect_error(mm[1, 4] <- 0.0)
})
