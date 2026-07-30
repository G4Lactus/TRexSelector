# =============================================================================
# Control parameter constructor functions for TRex selector variants.
#
# These mirror the Cpp struct hierarchy:
#   trex_control()        -> TRexControlParameter
#   trex_da_control()     -> TRexDAControlParameter
#   trex_gvs_control()    -> GVS control struct
#   trex_screen_control() -> ScreenTRexControlParameter
#   trex_biobank_control()-> BiobankScreenTRex FDR bounds
# =============================================================================

#' @name trex_control
#'
#' @title TRex Control Parameters
#'
#' @description Build a control list for TRex selector algorithms.
#'
#' @note **Seed recommendation for Monte Carlo FDR simulations:** always pass
#'   \code{seed = -1} (the default) to \code{TRexSelector$new()} or any other
#'   selector constructor when running repeated trials. A fixed integer seed
#'   causes every trial to draw K per-experiment dummy seeds as
#'   \code{mix_seed(fixed, 0), mix_seed(fixed, 1), \ldots} — deterministic but
#'   structurally correlated across trials — which can inflate the empirical FDR
#'   in a sweep. With \code{seed = -1} each of the K experiments independently
#'   draws hardware entropy via \code{std::random_device}, matching the
#'   behaviour of the standalone C++ demo.
#'
#' @param solver Solver algorithm: "TLARS", "TLASSO", "TENET", "TENET_AUG",
#'   "TSTAGEWISE", "TSTEPWISE", "TOMP", "TGP", "TACGP", "TMP", "TNCGMP",
#'   "TOOLS", "TAFS", "TCCD", "TCENET" (default: "TLARS"). "TENET_AUG" is
#'   the row-augmented formulation of the TENET elastic net (proved
#'   equivalent for \code{lambda2 >= 0}) and is accepted wherever TENET
#'   runs; it solves with the standard LASSO inner path unless the optional
#'   \code{tenet_aug_use_lars} is set. The CCD family (TCCD/TCENET)
#'   records exact penalized-lasso minimizers per dummy crossing. The
#'   IEN-family names "TIENET", "TIENET_AUG", and "TCIENET" are accepted by
#'   the parser but only dispatchable inside \code{TRexGVSSelector} with
#'   \code{gvs_type = "IEN"}; the plain selector rejects them with an
#'   informative error.
#' @param K Number of random experiments (default: 20).
#' @param tloop_stagnation_stop Whether to stop T-loop on stagnation. NULL (default) enables
#'   auto-detection: FALSE for non-greedy solvers (TLARS, TLASSO, TENET,
#'   TENET_AUG, and the CCD family TCCD/TCENET/TCIENET), TRUE for all others.
#' @param tloop_max_stagnant_steps Steps before stagnation trigger (default: 5).
#' @param use_openmp Whether to use OpenMP parallelization (default: FALSE).
#' @param use_memory_mapping Use out-of-core memory mapping (default: FALSE).
#' @param max_outer_threads Thread count for K repetitions (default: 1).
#' @param max_inner_threads Thread count for inner solvers (default: 1).
#' @param max_dummy_multiplier Maximum dummy features multiplier (default: 10).
#' @param use_max_T_stop Whether to bound max T_stop (default: TRUE).
#' @param opt_threshold Optimization threshold (default: 0.75).
#' @param lloop_strategy L-loop strategy: "SKIPL", "STANDARD", "HCONCAT",
#'   "PERMUTATION", "PERMUTATION_SEEDED", or "SEEDED" (default: "HCONCAT").
#'   The SEEDED variants re-derive dummies from the seed at every step
#'   (nothing stored); "PERMUTATION_SEEDED" is the stateless twin of
#'   "PERMUTATION" and produces identical experiments for the same seed.
#' @param tol Tolerance for solver algorithms (default: 1e-6).
#' @param lambda2 Elastic-net L2 penalty (TENET hyperparameter, default: 0.1).
#' @param rho_afs AFS hyperparameter (default: 0.3).
#' @param ncgmp_variant NCGMP variant: 0 = line_search, 1 = fully_corrective (default: 0).
#' @param exch_tie_alpha Exchangeable-tie band width for greedy solvers
#'   (TOMP/TAFS), in units of the pairwise ranking-noise sd. When > 0,
#'   statistically indistinguishable top candidates within highly correlated
#'   clusters are picked uniformly at random per step, restoring the
#'   within-experiment occurrence spread that the dependency-aware (DA)
#'   deflation's FDR control relies on. Recommended for greedy solvers under
#'   trex+DA: 0.25. \code{0} (default) turns the feature off (exact legacy
#'   behavior; ignored by path solvers).
#' @param exch_tie_floor Minimum absolute correlation for exchangeable-tie
#'   candidates, in (0, 1) (default: 0.5). Ignored unless
#'   \code{exch_tie_alpha > 0}.
#' @param tenet_aug_use_lars Logical, for \code{solver = "TENET_AUG"} in the
#'   plain/DA/screening selectors. The default \code{FALSE} runs the
#'   STANDARD LASSO inner path (variables can be dropped); \code{TRUE}
#'   opts into the pure-LARS inner path that never drops variables. Inside
#'   \code{TRexGVSSelector} use the \code{gvs_control} knob of the same
#'   name instead. Ignored by every other solver.
#' @param cd_lambda_rel_tol Relative lambda step tolerance of the CCD family's
#'   crossing search (default: 1e-5). Values <= 0 keep the solver's internal
#'   default. Ignored by the non-CCD solvers.
#' @param cd_tol_certify KKT certification tolerance of the CCD family. Must
#'   be set together with \code{cd_tol_probe}; the default -1 keeps the
#'   solver's own tolerances. Ignored by the non-CCD solvers.
#' @param cd_tol_probe Probing tolerance of the CCD family's coordinate
#'   sweeps. Must be set together with \code{cd_tol_certify}; the default -1
#'   keeps the solver's own tolerances. Ignored by the non-CCD solvers.
#' @param cd_gram_cap Column cap for the CCD family's cached Gram matrix;
#'   above it the solver falls back to the naive updates (default: 0 = keep
#'   the solver's internal default). Ignored by the non-CCD solvers.
#' @param cd_max_sweeps Maximum coordinate sweeps per CCD subproblem
#'   (default: 0 = keep the solver's internal default). Ignored by the
#'   non-CCD solvers.
#' @param dummy_distribution Dummy variable distribution, created with
#'   \code{\link{trex_dummy_distribution}} (default: standard Normal).
#'
#' @return A named list suitable for passing to TRex selector constructors.
#'
#' @examples
#' trex_control()
#' trex_control(solver = "TLASSO", K = 10)
#'
#' @export
trex_control <- function(solver = "TLARS",
                         K = 20,
                         tloop_stagnation_stop = NULL,
                         tloop_max_stagnant_steps = 5,
                         use_openmp = FALSE,
                         use_memory_mapping = FALSE,
                         max_outer_threads = 1,
                         max_inner_threads = 1,
                         max_dummy_multiplier = 10,
                         use_max_T_stop = TRUE,
                         opt_threshold = 0.75,
                         lloop_strategy = "HCONCAT",
                         tol = 1e-6,
                         lambda2 = 0.1,
                         rho_afs = 0.3,
                         ncgmp_variant = 0,
                         exch_tie_alpha = 0,
                         exch_tie_floor = 0.5,
                         tenet_aug_use_lars = FALSE,
                         cd_lambda_rel_tol = 1e-5,
                         cd_tol_certify = -1,
                         cd_tol_probe = -1,
                         cd_gram_cap = 0,
                         cd_max_sweeps = 0,
                         dummy_distribution = trex_dummy_distribution()) {

  if (max_dummy_multiplier < 1) {
    stop("max_dummy_multiplier must be >= 1. Got: ", max_dummy_multiplier)
  }
  if (tloop_max_stagnant_steps < 0) {
    stop("tloop_max_stagnant_steps must be >= 0. Got: ", tloop_max_stagnant_steps)
  }
  if (max_outer_threads < 1) {
    stop("max_outer_threads must be >= 1. Got: ", max_outer_threads)
  }
  if (max_inner_threads < 1) {
    stop("max_inner_threads must be >= 1. Got: ", max_inner_threads)
  }
  # The C++ side applies the certify/probe pair only when BOTH are > 0;
  # fail loudly here instead of silently ignoring a half-set pair.
  if (xor(cd_tol_certify > 0, cd_tol_probe > 0)) {
    stop("cd_tol_certify and cd_tol_probe must be set together (both > 0), ",
         "or both left at their defaults. Got: cd_tol_certify = ",
         cd_tol_certify, ", cd_tol_probe = ", cd_tol_probe)
  }
  if (cd_gram_cap < 0) {
    stop("cd_gram_cap must be >= 0. Got: ", cd_gram_cap)
  }
  if (cd_max_sweeps < 0) {
    stop("cd_max_sweeps must be >= 0. Got: ", cd_max_sweeps)
  }

  non_greedy_solvers <- c("TLARS", "TLASSO", "TENET", "TENET_AUG",
                          "TCCD", "TCENET", "TCIENET")
  if (is.null(tloop_stagnation_stop)) {
    tloop_stagnation_stop <- !(solver %in% non_greedy_solvers)
  }

  list(
    solver                  = solver,
    K                       = K,
    max_dummy_multiplier    = max_dummy_multiplier,
    use_max_T_stop          = use_max_T_stop,
    opt_threshold           = opt_threshold,
    lloop_strategy          = lloop_strategy,
    tloop_stagnation_stop   = tloop_stagnation_stop,
    tloop_max_stagnant_steps = tloop_max_stagnant_steps,
    parallel_rnd_experiments = use_openmp,
    use_memory_mapping      = use_memory_mapping,
    max_outer_threads       = max_outer_threads,
    max_inner_threads       = max_inner_threads,
    tol                     = tol,
    lambda2                 = lambda2,
    rho_afs                 = rho_afs,
    ncgmp_variant           = ncgmp_variant,
    exch_tie_alpha          = exch_tie_alpha,
    exch_tie_floor          = exch_tie_floor,
    tenet_aug_use_lars      = tenet_aug_use_lars,
    cd_lambda_rel_tol       = cd_lambda_rel_tol,
    cd_tol_certify          = cd_tol_certify,
    cd_tol_probe            = cd_tol_probe,
    cd_gram_cap             = cd_gram_cap,
    cd_max_sweeps           = cd_max_sweeps,
    dummy_distribution      = dummy_distribution
  )
}


#' @name trex_da_control
#'
#' @title TRex DA Control Parameters
#'
#' @description Build a control list for the dependency-aware dummy generation in TRexDASelector.
#'
#' @param da_method Dependency estimation method: "BT", "AR1", "EQUI", "NN"
#'   (default: "BT"). "NN" is a nearest-neighbour correlation-threshold sweep
#'   (BT-style) over a rho grid of length \code{hc_grid_length}.
#' @param cor_coef Correlation coefficient for AR1/EQUI. -2.0 triggers auto-estimation
#'   (default: -2.0).
#' @param rho_thr_DA Threshold below which equi-correlation is skipped (default: 0.02).
#' @param hc_linkage Hierarchical clustering linkage: "Single", "Complete", "Average",
#'   "WPGMA", "Ward" (default: "Single").
#' @param hc_grid_length Number of grid points for BT/NN calibration. \code{0}
#'   (default) lets the core auto-select \code{min(20, p)}.
#' @param prior_groups Optional prior grouping constraints: a list of integer
#'   vectors, each of length \code{p}, giving non-negative group labels per
#'   variable. When supplied (non-\code{NULL}, non-empty), the prior-groups
#'   dependency-aware path is used and \code{da_method} is ignored: the finest
#'   common refinement of all supplied levels acts as a hard merge constraint,
#'   hierarchical clustering (\code{hc_linkage}, correlation distance) runs
#'   within each constraint group, and the calibration rho grid (length
#'   \code{hc_grid_length}) is built from the pooled within-group dendrogram
#'   heights plus the conservative rho = 1 singleton anchor. The deflation
#'   therefore acts on tight data-driven neighbourhoods inside the known
#'   groups, never on the raw groups themselves. (default: \code{NULL}).
#'
#' @return A named list for use in TRexDASelector.
#'
#' @examples
#' trex_da_control()
#' trex_da_control(da_method = "BT", hc_linkage = "Average")
#' # Prior-groups path (constraints over p = 4 variables):
#' trex_da_control(prior_groups = list(c(1, 1, 2, 2)))
#'
#' @export
trex_da_control <- function(da_method = "BT",
                            cor_coef = -2.0,
                            rho_thr_DA = 0.02,
                            hc_linkage = "Single",
                            hc_grid_length = 0,
                            prior_groups = NULL) {
  list(
    da_method       = da_method,
    cor_coef        = cor_coef,
    rho_thr_DA      = rho_thr_DA,
    hc_linkage      = hc_linkage,
    hc_grid_length  = hc_grid_length,
    prior_groups    = prior_groups
  )
}


#' @name trex_gvs_control
#'
#' @title TRex GVS Control Parameters
#'
#' @description Build a control list for group variable selection augmentation policy in
#'   TRexGVSSelector. The solver is determined automatically from \code{gvs_type}:
#'   "EN" selects the T-Elastic-Net solver via \code{en_solver} — TENET
#'   (default), with TENET_AUG and TCENET as optional variants; "IEN" honors
#'   an IEN-family \code{solver} field in \code{\link{trex_control}} —
#'   TIENET (default, native pathwise), with TIENET_AUG and TCIENET as
#'   optional variants. For \code{gvs_type = "EN"} the \code{solver} field
#'   is ignored.
#'
#' @param gvs_type Augmentation policy: "EN" (Elastic Net) or "IEN" (Informed Elastic Net)
#'   (default: "EN").
#' @param corr_max Maximum pairwise correlation for automatic cluster formation (default: 0.5).
#' @param hc_linkage Hierarchical clustering linkage: "Single", "Complete", "Average",
#'   "WPGMA" (default: "Single").
#' @param lambda_2 L2 penalty weight for group structure, in LARS units.
#'   A negative value (default \code{-1}) triggers automatic determination via
#'   the method selected by \code{lambda2_method}; \code{0} is the degenerate
#'   pure T-LASSO case (no ridge); a positive value is used as-is.
#' @param lambda2_method Auto-determination strategy for \code{lambda_2} when
#'   \code{lambda_2 < 0}. \code{NULL} (default) resolves by track:
#'   \code{"CV_1SE_CCD"} for \code{gvs_type = "EN"} and
#'   \code{"CV_1SE_IEN_CCD"} for \code{gvs_type = "IEN"} (the IEN-geometry
#'   tuner consumes the group structure; the 1SE rule is preferred because
#'   the MIN rule tends to pin to the grid floor). Explicit choices:
#'   EN-shaped ridge CV \code{"CV_1SE_CCD"}, \code{"CV_MIN_CCD"},
#'   \code{"CV_1SE_SVD"}, \code{"CV_MIN_SVD"} (allowed on both tracks);
#'   IEN-geometry \code{"CV_1SE_IEN_CCD"}, \code{"CV_MIN_IEN_CCD"},
#'   \code{"CV_1SE_TIK_SVD"}, \code{"CV_MIN_TIK_SVD"} (require
#'   \code{gvs_type = "IEN"} — they consume the group structure).
#' @param en_solver Elastic Net solver variant for \code{gvs_type = "EN"}:
#'   \code{"TENET"} (default, Gram-based), \code{"TENET_AUG"}
#'   (row-augmented LASSO), or \code{"TCENET"} (CCD; exact fixed-lambda1
#'   elastic-net minimizers per dummy crossing). All solve the same EN
#'   problem and select the same variables up to path-vs-minimizer
#'   differences; the switch exists to validate solver equivalence,
#'   \strong{not} to choose a LARS vs LASSO path (for that, see
#'   \code{tenet_aug_use_lars}).
#' @param tenet_aug_use_lars Logical. When \code{en_solver = "TENET_AUG"}, run
#'   the augmented system with a pure-LARS inner path that never drops variables
#'   (\code{TRUE}) instead of the default LASSO inner path that can
#'   (\code{FALSE}, default). This is the LARS-vs-LASSO distinction. Ignored for
#'   \code{en_solver = "TENET"}.
#' @param groups Integer vector of manually specified group assignments (1-based, default: NULL).
#' @param cv_n_folds Number of folds for the k-fold cross-validation used by the
#'   CV \code{lambda2_method} paths (default: 10, matching \code{cv.glmnet}).
#'   Ignored when \code{lambda_2 >= 0}.
#' @param cv_n_lambda Number of points on the geometric lambda grid searched by
#'   the CV auto-computation of \code{lambda_2} (default: 1000). Ignored when
#'   \code{lambda_2 >= 0}.
#' @param cv_seed Seed for the deterministic fold-permutation RNG used by the CV
#'   \code{lambda2_method} paths. \code{-1} (default) derives it from the T-Rex
#'   \code{seed} (deterministic when \code{seed >= 0}, hardware entropy when
#'   \code{seed < 0}); a value \code{>= 0} overrides \code{seed} for CV folds
#'   only, giving per-trial fold reproducibility.
#'
#' @return A named list for use in TRexGVSSelector.
#'
#' @examples
#' trex_gvs_control()
#' trex_gvs_control(gvs_type = "IEN", corr_max = 0.3)
#'
#' @export
trex_gvs_control <- function(gvs_type = "EN",
                             corr_max = 0.5,
                             hc_linkage = "Single",
                             lambda_2 = -1.0,
                             lambda2_method = NULL,
                             en_solver = "TENET",
                             tenet_aug_use_lars = FALSE,
                             groups = NULL,
                             cv_n_folds = 10,
                             cv_n_lambda = 1000,
                             cv_seed = -1) {
  # NULL resolves by track: the EN-shaped ridge CV for EN, the IEN-geometry
  # tuner (group-consuming, no p/2 conversion) for IEN. 1SE in both cases —
  # the MIN rule tends to pin to the grid floor.
  if (is.null(lambda2_method)) {
    lambda2_method <- if (gvs_type == "IEN") "CV_1SE_IEN_CCD" else "CV_1SE_CCD"
  }
  lambda2_method <- match.arg(lambda2_method,
                              c("CV_1SE_CCD", "CV_MIN_CCD",
                                "CV_1SE_SVD", "CV_MIN_SVD",
                                "CV_1SE_IEN_CCD", "CV_MIN_IEN_CCD",
                                "CV_1SE_TIK_SVD", "CV_MIN_TIK_SVD"))
  ien_methods <- c("CV_1SE_IEN_CCD", "CV_MIN_IEN_CCD",
                   "CV_1SE_TIK_SVD", "CV_MIN_TIK_SVD")
  if (gvs_type == "EN" && lambda2_method %in% ien_methods) {
    stop("lambda2_method = \"", lambda2_method, "\" is an IEN-geometry ",
         "tuner (it consumes the group structure) and requires ",
         "gvs_type = \"IEN\".")
  }
  en_solver <- match.arg(en_solver, c("TENET", "TENET_AUG", "TCENET"))
  list(
    gvs_type           = gvs_type,
    corr_max           = corr_max,
    hc_linkage         = hc_linkage,
    lambda_2           = lambda_2,
    lambda2_method     = lambda2_method,
    en_solver          = en_solver,
    tenet_aug_use_lars = as.logical(tenet_aug_use_lars),
    groups             = groups,
    cv_n_folds         = as.integer(cv_n_folds),
    cv_n_lambda        = as.integer(cv_n_lambda),
    cv_seed            = as.integer(cv_seed)
  )
}


#' @name trex_screen_control
#'
#' @title TRex Screen Control Parameters
#'
#' @description Build a control list for the screening step in TRexScreeningSelector and
#'   TRexBiobankScreeningSelector.
#'
#' @param use_bootstrap_CI Whether to use bootstrap confidence intervals for screening
#'   (default: FALSE).
#' @param R_boot Number of bootstrap iterations (default: 1000).
#' @param ci_grid_step Grid step size for confidence-interval calculations (default: 0.001).
#' @param trex_method Screening method: "TREX", "TREX_DA_AR1", "TREX_DA_EQUI",
#'   "TREX_DA_BLOCK_EQUI" (default: "TREX").
#'
#' @return A named list for use in TRexScreeningSelector or TRexBiobankScreeningSelector.
#'
#' @examples
#' trex_screen_control()
#' trex_screen_control(use_bootstrap_CI = TRUE, R_boot = 500)
#'
#' @export
trex_screen_control <- function(use_bootstrap_CI = FALSE,
                                R_boot = 1000,
                                ci_grid_step = 0.001,
                                trex_method = "TREX") {
  list(
    use_bootstrap_CI = use_bootstrap_CI,
    R_boot           = R_boot,
    ci_grid_step     = ci_grid_step,
    trex_method      = trex_method
  )
}


#' @name trex_biobank_control
#'
#' @title TRex Biobank Control Parameters
#'
#' @description Build a control list for the FDR acceptance bounds in
#'   TRexBiobankScreeningSelector. The target FDR level (\code{tFDR}) is a separate
#'   top-level parameter of the selector constructor.
#'
#' @param lower_bound_FDR Lower bound of the acceptable estimated FDR range (default: 0.05).
#' @param upper_bound_FDR Upper bound of the acceptable estimated FDR range (default: 0.15).
#'
#' @return A named list for use in TRexBiobankScreeningSelector.
#'
#' @examples
#' trex_biobank_control()
#' trex_biobank_control(lower_bound_FDR = 0.02, upper_bound_FDR = 0.10)
#'
#' @export
trex_biobank_control <- function(lower_bound_FDR = 0.05,
                                 upper_bound_FDR = 0.15) {
  list(
    lower_bound_FDR = lower_bound_FDR,
    upper_bound_FDR = upper_bound_FDR
  )
}


#' @name trex_dummy_distribution
#'
#' @title TRex Dummy Variable Distribution
#'
#' @description Build a distribution specification for the dummy variables
#'   injected by the T-Rex Selector. Pass the result to the
#'   \code{dummy_distribution} argument of \code{\link{trex_control}}.
#'
#' @param type Character string naming the distribution.  One of:
#'   \code{"Normal"}, \code{"Uniform"}, \code{"Rademacher"},
#'   \code{"StudentT"}, \code{"Laplace"}, \code{"Gumbel"},
#'   \code{"Triangle"}, \code{"UniformSphere"},
#'   \code{"Mammen"}, \code{"ConstrainedSparseRademacher"}, \code{"Logistic"}.
#'   Defaults to \code{"Normal"}.
#' @param a   Scale / upper-bound parameter (used by \code{"Uniform"}).
#' @param df  Degrees of freedom (used by \code{"StudentT"}).
#' @param location  Location parameter (used by \code{"Laplace"},
#'   \code{"Gumbel"}, \code{"Logistic"}).
#' @param scale  Scale parameter (used by \code{"Laplace"},
#'   \code{"Gumbel"}, \code{"Logistic"}).
#' @param b  Upper vertex of the triangle (used by \code{"Triangle"}).
#' @param c  Right boundary of the triangle (used by \code{"Triangle"}).
#' @param dim  Dimension of the sphere (used by \code{"UniformSphere"}).
#' @param s  Sparsity parameter (used by \code{"ConstrainedSparseRademacher"}).
#'
#' @return A named list describing the dummy distribution, for use in
#'   \code{\link{trex_control}}.
#'
#' @examples
#' # Standard Normal (default)
#' trex_dummy_distribution()
#'
#' # Student-t with 3 degrees of freedom
#' trex_dummy_distribution("StudentT", df = 3)
#'
#' @export
trex_dummy_distribution <- function(type     = "Normal",
                                    a        = NULL,
                                    df       = NULL,
                                    location = NULL,
                                    scale    = NULL,
                                    b        = NULL,
                                    c        = NULL,
                                    dim      = NULL,
                                    s        = NULL) {
  valid_types <- c("Normal", "Uniform", "Rademacher", "StudentT",
                   "Laplace", "Gumbel", "Triangle",
                   "UniformSphere", "Mammen", "ConstrainedSparseRademacher",
                   "Logistic")
  if (!type %in% valid_types) {
    stop("Unknown distribution type: '", type, "'. ",
         "Must be one of: ", paste(valid_types, collapse = ", "), ".")
  }
  dist <- list(type = type)
  if (!is.null(a))        dist$a        <- a
  if (!is.null(df))       dist$df       <- df
  if (!is.null(location)) dist$location <- location
  if (!is.null(scale))    dist$scale    <- scale
  if (!is.null(b))        dist$b        <- b
  if (!is.null(c))        dist$c        <- c
  if (!is.null(dim))      dist$dim      <- dim
  if (!is.null(s))        dist$s        <- s
  dist
}
