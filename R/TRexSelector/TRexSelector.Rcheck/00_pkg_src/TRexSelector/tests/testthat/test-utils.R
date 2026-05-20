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
