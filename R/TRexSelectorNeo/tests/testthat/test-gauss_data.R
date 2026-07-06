# =============================================================================
# test-gauss_data.R
#
# Integration tests using the bundled Gauss_data dataset (n = 50, p = 100,
# 3 true signals at indices 1-3).
#
# Test groups
#   1. All TSolver variants — execute with T_stop = 3
#   2. All TRex selector methods — select with tFDR = 0.10
# =============================================================================

data(Gauss_data)

X_g <- Gauss_data$X
y_g <- as.vector(Gauss_data$y)
p_g <- ncol(X_g)
n_g <- nrow(X_g)

set.seed(0)
D_g <- matrix(rnorm(n_g * p_g), n_g, p_g)

# =============================================================================
# Group 1: All TSolver variants on Gauss_data, T_stop = 3
# =============================================================================

.gauss_factories <- list(
  TLARS      = function() TLARS_Solver$new(X_g, D_g, y_g),
  TLASSO     = function() TLASSO_Solver$new(X_g, D_g, y_g),
  TSTEPWISE  = function() TSTEPWISE_Solver$new(X_g, D_g, y_g),
  TENET      = function() TENET_Solver$new(X_g, D_g, y_g, lambda2 = 0.5),
  TSTAGEWISE = function() TSTAGEWISE_Solver$new(X_g, D_g, y_g),
  TOMP       = function() TOMP_Solver$new(X_g, D_g, y_g),
  TGP        = function() TGP_Solver$new(X_g, D_g, y_g),
  TACGP      = function() TACGP_Solver$new(X_g, D_g, y_g),
  TMP        = function() TMP_Solver$new(X_g, D_g, y_g),
  TOOLS      = function() TOOLS_Solver$new(X_g, D_g, y_g),
  TNCGMP_ls  = function() TNCGMP_Solver$new(X_g, D_g, y_g, variant = "line_search"),
  TNCGMP_fc  = function() TNCGMP_Solver$new(X_g, D_g, y_g, variant = "fully_corrective"),
  TAFS       = function() TAFS_Solver$new(X_g, D_g, y_g, rho = 1.0)
)

test_that("All TSolver variants execute on Gauss_data with T_stop = 3", {
  for (nm in names(.gauss_factories)) {
    s <- .gauss_factories[[nm]]()
    expect_no_error(s$execute_step(T_stop = 3))
    expect_gte(s$get_num_steps(), 1L)
    rss <- s$get_rss()
    expect_gte(length(rss), 1L)
    expect_true(all(is.finite(rss)))
  }
})


# =============================================================================
# Group 2: All TRex selector methods on Gauss_data, tFDR = 0.10
# =============================================================================

test_that("TRexSelector runs on Gauss_data and returns valid selection", {
  sel <- TRexSelector$new(X_g, y_g, tFDR = 0.10, verbose = FALSE, control = trex_control(K = 20))
  sel$select()

  idx <- sel$selected_indices
  sv  <- sel$selected_var

  expect_type(idx, "integer")
  expect_length(sv, p_g)
  if (length(idx) > 0) {
    expect_true(all(idx >= 1L & idx <= p_g))
  }
  expect_equal(sort(idx), which(sv))
})


test_that("TRexGVSSelector (EN and IEN) runs on Gauss_data and returns valid selection", {
  for (gvs in c("EN", "IEN")) {
    sel <- TRexGVSSelector$new(X_g, y_g, tFDR = 0.10, verbose = FALSE,
                               gvs_control = trex_gvs_control(gvs_type = gvs),
                               control = trex_control(K = 20))
    sel$select()

    idx <- sel$selected_indices
    sv  <- sel$selected_var

    expect_type(idx, "integer")
    expect_length(sv, p_g)
    if (length(idx) > 0) {
      expect_true(all(idx >= 1L & idx <= p_g))
    }
    expect_equal(sort(idx), which(sv))
  }
})


test_that("TRexScreeningSelector runs on Gauss_data and returns valid selection", {
  sel <- TRexScreeningSelector$new(X_g, y_g, verbose = FALSE, control = trex_control(K = 20))
  sel$select()

  idx <- sel$selected_indices
  sv  <- sel$selected_var

  expect_type(idx, "integer")
  expect_length(sv, p_g)
  if (length(idx) > 0) {
    expect_true(all(idx >= 1L & idx <= p_g))
  }
  expect_equal(sort(idx), which(sv))
})


test_that("TRexDASelector runs on Gauss_data and returns valid selection", {
  sel <- TRexDASelector$new(X_g, y_g, tFDR = 0.10, verbose = FALSE, control = trex_control(K = 20))
  sel$select()

  idx <- sel$selected_indices
  sv  <- sel$selected_var

  expect_type(idx, "integer")
  expect_length(sv, p_g)
  if (length(idx) > 0) {
    expect_true(all(idx >= 1L & idx <= p_g))
  }
  expect_equal(sort(idx), which(sv))
})


test_that("TRexBiobankScreeningSelector runs on Gauss_data and returns valid selection", {
  sel <- TRexBiobankScreeningSelector$new(X_g, y_g, tFDR = 0.10,
                                          verbose = FALSE, control = trex_control(K = 20))
  res <- sel$select()

  idx <- res$selected_indices
  expect_type(idx, "integer")
  if (length(idx) > 0) {
    expect_true(all(idx >= 1L & idx <= p_g))
  }
})
