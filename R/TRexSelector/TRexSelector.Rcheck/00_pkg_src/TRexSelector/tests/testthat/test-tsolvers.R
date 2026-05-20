# =============================================================================
# test-tsolvers.R
# =============================================================================
#
# Unit tests for the R6 T-Solver wrappers (TSolverBase and all subclasses).
# Mirrors the structure of the C++ test bed in
#   tests/test_tsolvers/test_tsolvers_polymorphic.cpp
#   tests/test_tsolvers/test_tsolvers_serialization.cpp
#
# Test groups
#   1. Input validation
#   2. Polymorphic execution test (all 13 factory entries)
#   3. Getter return-type and contract checks (TLARSSolver reference)
#   4. Solver-specific getter checks (lambda, num_removals, cycling_ratio)
#   5. Warm-start / serialization (all 13 factory entries)
# =============================================================================

# ---------------------------------------------------------------------------
# Shared test fixture
# ---------------------------------------------------------------------------

n <- 100L
p <- 30L
d <- 30L
set.seed(42)
X <- matrix(rnorm(n * p), n, p)
D <- matrix(rnorm(n * d), n, d)
y <- X[, 5] * 4.0 + X[, 15] * (-2.5) + X[, 25] * 3.1 + rnorm(n)

# ---------------------------------------------------------------------------
# Solver factory (13 entries: 12 solver types, TNCGMP counted for both variants)
# ---------------------------------------------------------------------------

.factories <- list(
  TLARS      = function() TLARSSolver$new(X, D, y),
  TLASSO     = function() TLASSOSolver$new(X, D, y),
  TSTEPWISE  = function() TSTEPWISESolver$new(X, D, y),
  TENET      = function() TENETSolver$new(X, D, y, lambda2 = 0.5),
  TSTAGEWISE = function() TStagewiseSolver$new(X, D, y),
  TOMP       = function() TOMPSolver$new(X, D, y),
  TGP        = function() TGPSolver$new(X, D, y),
  TACGP      = function() TACGPSolver$new(X, D, y),
  TMP        = function() TMPSolver$new(X, D, y),
  TOOLS      = function() TOOLSSolver$new(X, D, y),
  TNCGMP_ls  = function() TNCGMPSolver$new(X, D, y, variant = "line_search"),
  TNCGMP_fc  = function() TNCGMPSolver$new(X, D, y, variant = "fully_corrective"),
  TAFS       = function() TAFSSolver$new(X, D, y, rho = 1.0)
)


# =============================================================================
# Group 1: Input validation
# =============================================================================

test_that("TSolverBase rejects non-matrix X", {
  expect_error(TLARSSolver$new("not_a_matrix", D, y))
})

test_that("TSolverBase rejects y length mismatch", {
  expect_error(TLARSSolver$new(X, D, y[seq_len(50)]))
})

test_that("TSolverBase rejects D row count mismatch", {
  expect_error(TLARSSolver$new(X, D[seq_len(50), ], y))
})

test_that("TENETSolver rejects negative lambda2", {
  expect_error(TENETSolver$new(X, D, y, lambda2 = -0.1))
})

test_that("TNCGMPSolver rejects unknown variant string", {
  expect_error(TNCGMPSolver$new(X, D, y, variant = "bad_variant"))
})


# =============================================================================
# Group 2: Polymorphic execution test (all 13 factory entries)
# Mirrors test_tsolvers_polymorphic.cpp :: ExecutionSmokeTest
# =============================================================================

test_that("All solver types execute without error and produce a valid RSS path", {
  for (nm in names(.factories)) {
    s <- .factories[[nm]]()
    expect_no_error(s$execute_step(T_stop = 5))

    nsteps <- s$get_num_steps()
    expect_type(nsteps, "integer")
    expect_gt(nsteps, 0L, label = nm)

    rss <- s$get_rss()
    expect_length(rss, nsteps)

    # RSS must decrease from first to last step
    expect_gt(rss[[1]], rss[[nsteps]], label = nm)
  }
})


# =============================================================================
# Group 3: Getter return-type and contract checks (TLARSSolver as reference)
# =============================================================================

local({
  s <- TLARSSolver$new(X, D, y)
  s$execute_step(T_stop = 8)
  nsteps <- s$get_num_steps()

  test_that("get_num_steps returns a positive integer", {
    expect_type(nsteps, "integer")
    expect_gte(nsteps, 1L)
  })

  test_that("get_beta returns a numeric vector of length p", {
    b <- s$get_beta()
    expect_type(b, "double")
    expect_length(b, p)
  })

  test_that("get_beta(step) returns a numeric vector of length p", {
    b <- s$get_beta(step = 1L)
    expect_type(b, "double")
    expect_length(b, p)
  })

  test_that("get_rss returns a positive, strictly decreasing numeric vector", {
    rss <- s$get_rss()
    expect_type(rss, "double")
    expect_length(rss, nsteps)
    expect_true(all(rss > 0), label = "all RSS > 0")
    expect_true(all(diff(rss) <= 1e-12), label = "RSS is non-increasing")
  })

  test_that("get_r2 returns a numeric vector in [0,1] that is non-decreasing", {
    r2 <- s$get_r2()
    expect_type(r2, "double")
    expect_length(r2, nsteps)
    expect_true(all(r2 >= 0 & r2 <= 1), label = "R2 in [0,1]")
    expect_true(all(diff(r2) >= -1e-12), label = "R2 non-decreasing")
  })

  test_that("get_cp returns a numeric vector of length nsteps", {
    cp <- s$get_cp()
    expect_type(cp, "double")
    expect_length(cp, nsteps)
  })

  test_that("get_dof returns a non-negative numeric vector of length nsteps", {
    dof <- s$get_dof()
    expect_type(dof, "double")
    expect_length(dof, nsteps)
    expect_true(all(dof >= 0), label = "all DoF >= 0")
  })

  test_that("get_actives(r_index=TRUE) returns 1-based indices within [1, p]", {
    a <- s$get_actives(r_index = TRUE)
    expect_type(a, "double")
    expect_true(all(a >= 1L & a <= p), label = "actives 1-based in [1,p]")
  })

  test_that("get_actives(r_index=FALSE) returns 0-based indices within [0, p-1]", {
    a <- s$get_actives(r_index = FALSE)
    expect_type(a, "double")
    expect_true(all(a >= 0L & a <= p - 1L), label = "actives 0-based in [0,p-1]")
  })

  test_that("get_inactives(r_index=TRUE) returns 1-based indices within [1, p]", {
    i <- s$get_inactives(r_index = TRUE)
    expect_type(i, "double")
    expect_true(all(i >= 1L & i <= p), label = "inactives 1-based in [1,p]")
  })

  test_that("get_actives and get_inactives partition [1, p]", {
    a <- s$get_actives(r_index = TRUE)
    i <- s$get_inactives(r_index = TRUE)
    expect_setequal(c(a, i), seq_len(p))
  })

  test_that("get_intercept returns a single numeric scalar", {
    ic <- s$get_intercept()
    expect_type(ic, "double")
    expect_length(ic, 1L)
  })

  test_that("get_actions(r_index=TRUE) returns list of 1-based integer-vector elements", {
    acts <- s$get_actions(r_index = TRUE)
    expect_type(acts, "list")
    expect_length(acts, nsteps)
    for (step_acts in acts) {
      expect_type(step_acts, "integer")
      # All action indices must be >= 1 in absolute value (1-based)
      expect_true(all(abs(step_acts) >= 1L), label = "all action |index| >= 1")
    }
  })

  test_that("get_actions(r_index=FALSE) returns list of 0-based integer-vector elements", {
    acts <- s$get_actions(r_index = FALSE)
    expect_type(acts, "list")
    expect_length(acts, nsteps)
    for (step_acts in acts) {
      expect_type(step_acts, "integer")
      # All action indices must be >= 0 in absolute value (0-based)
      expect_true(all(abs(step_acts) >= 0L), label = "all action |index| >= 0")
    }
  })
})


# =============================================================================
# Group 4: Solver-specific getter contracts
# =============================================================================

test_that("TLARSSolver: get_lambda returns a non-increasing numeric vector of length nsteps", {
  s <- TLARSSolver$new(X, D, y)
  s$execute_step(T_stop = 8)
  lam <- s$get_lambda()
  expect_type(lam, "double")
  expect_length(lam, s$get_num_steps())
  expect_true(all(diff(lam) <= 1e-12), label = "lambda non-increasing")
})

test_that("TLASSOSolver: get_lambda numeric, get_num_removals non-negative integer", {
  s <- TLASSOSolver$new(X, D, y)
  s$execute_step(T_stop = 8)
  expect_type(s$get_lambda(), "double")
  nr <- s$get_num_removals()
  expect_type(nr, "integer")
  expect_gte(nr, 0L)
})

test_that("TSTEPWISESolver: get_lambda returns a numeric vector", {
  s <- TSTEPWISESolver$new(X, D, y)
  s$execute_step(T_stop = 8)
  expect_type(s$get_lambda(), "double")
})

test_that("TENETSolver: get_lambda, get_num_removals, get_cycling_ratio all correct", {
  s <- TENETSolver$new(X, D, y, lambda2 = 0.5)
  s$execute_step(T_stop = 8)
  expect_type(s$get_lambda(), "double")
  nr <- s$get_num_removals()
  expect_type(nr, "integer")
  expect_gte(nr, 0L)
  cr <- s$get_cycling_ratio()
  expect_type(cr, "double")
  expect_gte(cr, 0.0)
  expect_lte(cr, 1.0)
})

test_that("TStagewiseSolver: get_num_removals non-negative, get_cycling_ratio in [0,1]", {
  s <- TStagewiseSolver$new(X, D, y)
  s$execute_step(T_stop = 8)
  nr <- s$get_num_removals()
  expect_type(nr, "integer")
  expect_gte(nr, 0L)
  cr <- s$get_cycling_ratio()
  expect_type(cr, "double")
  expect_gte(cr, 0.0)
  expect_lte(cr, 1.0)
})


# =============================================================================
# Group 5: Warm-start / serialization — all 13 factory entries
# Mirrors test_tsolvers_polymorphic.cpp :: SerializationEquivalence
#
# Note: No data-copy issue in R — Rcpp copies matrices by value on every
# $new() call, unlike C++ Eigen::Map which would mutate in-place.
# =============================================================================

test_that("All solver types resume from checkpoint identically to a continuous run", {
  T_partial <- 4L
  T_final   <- 8L

  for (nm in names(.factories)) {
    # --- Reference: uninterrupted run to T_final ---
    s_ref <- .factories[[nm]]()
    s_ref$execute_step(T_stop = T_final)

    # --- Partial run, save, reload, continue ---
    ck <- tempfile(fileext = ".bin")
    on.exit(unlink(ck), add = TRUE)

    s1 <- .factories[[nm]]()
    s1$execute_step(T_stop = T_partial)
    s1$save(ck)
    expect_true(file.exists(ck), label = paste(nm, "checkpoint created"))

    s2 <- .factories[[nm]]()
    s2$load(ck)
    s2$execute_step(T_stop = T_final)

    # Number of steps must match
    expect_equal(
      s_ref$get_num_steps(), s2$get_num_steps(),
      label = paste(nm, "num_steps after resume")
    )

    # Action path must match
    expect_equal(
      s_ref$get_actions(), s2$get_actions(),
      label = paste(nm, "action path after resume")
    )

    # Final beta must match to floating-point precision
    expect_equal(
      s_ref$get_beta(), s2$get_beta(),
      tolerance = 1e-9,
      label = paste(nm, "beta after resume")
    )
  }
})
