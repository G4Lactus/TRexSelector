// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "TRexSelector_types.h"   // TSolverRcpp, the zero-copy holder
#include <fstream>
#include <memory>
#include <cereal/archives/portable_binary.hpp>
#include <tsolvers/ad/tsolver_base.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tstepwise_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tenet_aug_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tienet_aug_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tstagewise_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tomp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tgp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tacgp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tmp_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tools_solver.hpp>
#include <tsolvers/ad/linear_model/omp_based/tncgmp_solver.hpp>
#include <tsolvers/ad/linear_model/afs_based/tafs_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tienet_solver.hpp>
#include <tsolvers/ad/linear_model/lars_based/tiknet_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdlasso_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdenet_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdienet_solver.hpp>
#include <tsolvers/ad/linear_model/cd_based/tcdtiknet_solver.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>

// ==============================================================================

/*
 * Note on Roxygen Documentation:
 * All functions exported via Rcpp in this file are documented using standard Roxygen tags
 * but strictly include the `@noRd` tag. These are internal C++ bindings that are wrapped
 * by user-friendly R6 classes and R functions in the package namespace. Generating `.Rd`
 * manuals for these endpoints would clutter the public API.
 */

using namespace Rcpp;
using namespace trex::utils::memmap;
using namespace trex::tsolvers::ad;
using namespace trex::tsolvers::ad::linear_model::lars_based;
using namespace trex::tsolvers::ad::linear_model::omp_based;
using namespace trex::tsolvers::ad::linear_model::afs_based;
using namespace trex::tsolvers::ad::linear_model::cd_based;

// ==============================================================================
// TSolverRcpp — zero-copy holder, defined once in TRexSelector_types.h
// ==============================================================================
//
// The holder keeps the Eigen maps over the R matrices (or the mmap file) and
// owns the solver; RcppExports.cpp instantiates the XPtr finalizer from the
// same definition. See the header for the in-place contract.

/** @brief Map an R double matrix without copying. */
static Eigen::Map<Eigen::MatrixXd> map_of(Rcpp::NumericMatrix& m) {
    return {m.begin(), m.nrow(), m.ncol()};
}

/** @brief Map an R double vector without copying. */
static Eigen::Map<Eigen::VectorXd> map_of(Rcpp::NumericVector& v) {
    return {v.begin(), v.size()};
}

// ==============================================================================
// TSolver Instance Creators
// ==============================================================================
//
// Every solver is constructed as (X, D, y, options) — plus the 1-based group
// vector of the IEN solvers (TIENETAug, TIENET, TCDIENET) or the Tikhonov
// matrix of the sparse-K solvers (TIKNET, TCDTIKNET) — where
// `options` is the R list built by tsolver_options().
// parse_solver_options() maps it onto the C++ SolverOptions; absent fields
// keep the C++ defaults.

/**
 * @brief Parse the tsolver_options() list into trex::tsolvers::ad::SolverOptions.
 */
SolverOptions parse_solver_options(const Rcpp::List& o) {
    SolverOptions opts;
    auto has = [&](const char* name) { return o.containsElementNamed(name); };

    if (has("normalize")) opts.normalize = Rcpp::as<bool>(o["normalize"]);
    if (has("intercept")) opts.intercept = Rcpp::as<bool>(o["intercept"]);
    if (has("precentered")) opts.precentered = Rcpp::as<bool>(o["precentered"]);
    if (has("verbose"))   opts.verbose   = Rcpp::as<bool>(o["verbose"]);
    if (has("scaling_mode")) {
        const std::string m = Rcpp::as<std::string>(o["scaling_mode"]);
        if (m == "L2")          opts.scaling_mode = ScalingMode::L2;
        else if (m == "ZSCORE") opts.scaling_mode = ScalingMode::ZSCORE;
        else Rcpp::stop("tsolver_options: scaling_mode must be \"L2\" or \"ZSCORE\"; got " + m);
    }
    if (has("tol"))            opts.tol            = Rcpp::as<double>(o["tol"]);
    if (has("tie_seed"))       opts.tie_seed       = static_cast<long long>(Rcpp::as<double>(o["tie_seed"]));
    if (has("n_threads"))      opts.n_threads      = Rcpp::as<int>(o["n_threads"]);
    if (has("exch_tie_alpha")) opts.exch_tie_alpha = Rcpp::as<double>(o["exch_tie_alpha"]);
    if (has("exch_tie_floor")) opts.exch_tie_floor = Rcpp::as<double>(o["exch_tie_floor"]);
    if (has("lambda2"))        opts.lambda2        = Rcpp::as<double>(o["lambda2"]);
    if (has("rho_afs"))        opts.rho_afs        = Rcpp::as<double>(o["rho_afs"]);
    if (has("ncgmp_variant")) {
        const std::string v = Rcpp::as<std::string>(o["ncgmp_variant"]);
        if (v == "line_search")           opts.ncgmp_variant = NCGMPVariant::LineSearch;
        else if (v == "fully_corrective") opts.ncgmp_variant = NCGMPVariant::FullyCorrective;
        else Rcpp::stop("tsolver_options: ncgmp_variant must be \"line_search\" or \"fully_corrective\"; got " + v);
    }
    if (has("tenet_aug_use_lars")) opts.tenet_aug_use_lars = Rcpp::as<bool>(o["tenet_aug_use_lars"]);
    if (has("kappa_dummy"))        opts.kappa_dummy        = Rcpp::as<double>(o["kappa_dummy"]);
    if (has("fold_dummies"))       opts.fold_dummies       = Rcpp::as<bool>(o["fold_dummies"]);
    if (has("group_penalty")) {
        const std::string g = Rcpp::as<std::string>(o["group_penalty"]);
        if (g == "MEAN")           opts.group_penalty = GroupPenaltyVariant::MEAN;
        else if (g == "CENTERING") opts.group_penalty = GroupPenaltyVariant::CENTERING;
        else Rcpp::stop("tsolver_options: group_penalty must be \"MEAN\" or \"CENTERING\"; got " + g);
    }
    if (has("cd_lambda_rel_tol")) opts.cd_lambda_rel_tol = Rcpp::as<double>(o["cd_lambda_rel_tol"]);
    if (has("cd_tol_certify"))    opts.cd_tol_certify    = Rcpp::as<double>(o["cd_tol_certify"]);
    if (has("cd_tol_probe"))      opts.cd_tol_probe      = Rcpp::as<double>(o["cd_tol_probe"]);
    if (has("cd_gram_cap"))       opts.cd_gram_cap       = static_cast<std::size_t>(Rcpp::as<double>(o["cd_gram_cap"]));
    if (has("cd_max_sweeps")) {
        opts.cd_max_sweeps = static_cast<std::size_t>(Rcpp::as<double>(o["cd_max_sweeps"]));
    }
    if (has("cd_certified_localization"))
        opts.cd_certified_localization = Rcpp::as<bool>(o["cd_certified_localization"]);
    return opts;
}

/** @brief 1-based R group ids -> 0-based contiguous ids (the solver contract). */
static Eigen::VectorXi groups_from_r(const IntegerVector& groups) {
    Eigen::VectorXi c_groups(groups.size());
    for (int i = 0; i < groups.size(); ++i) c_groups[i] = groups[i] - 1;
    return c_groups;
}

/** @brief A Tikhonov matrix from R: a Matrix::dgCMatrix (S4) or a dense
 *  matrix (converted via sparseView()); same contract as the Tikhonov
 *  selector's tikhonov_K. */
static Eigen::SparseMatrix<double> tsolver_as_sparse_matrix(SEXP m) {
    if (Rf_isS4(m)) {
        return Rcpp::as<Eigen::SparseMatrix<double>>(m);
    }
    const Eigen::MatrixXd dense = Rcpp::as<Eigen::MatrixXd>(m);
    return dense.sparseView();
}

// ==============================================================================
// Instance creators (in-memory X)
// ==============================================================================

//' @title Create TLARS_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_lars_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TLARS_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TLASSO_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tlasso_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TLASSO_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TSTEPWISE_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tstepwise_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TSTEPWISE_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TSTAGEWISE_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tstagewise_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TSTAGEWISE_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TENET_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tenet_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TENET_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TENETAug_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tenet_aug_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TENETAug_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TIENETAug_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param groups 1-based group ids, one per predictor
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tienet_aug_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    IntegerVector groups,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TIENETAug_Solver(h->X_map_, h->D_map_, h->y_map_, groups_from_r(groups), opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TIENET_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param groups 1-based group ids, one per predictor
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tienet_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    IntegerVector groups,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TIENET_Solver(h->X_map_, h->D_map_, h->y_map_, groups_from_r(groups), opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TIKNET_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param tikhonov_K p x p PSD Tikhonov matrix (dgCMatrix or dense)
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tiknet_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    SEXP tikhonov_K,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TIKNET_Solver(h->X_map_, h->D_map_, h->y_map_, tsolver_as_sparse_matrix(tikhonov_K), opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TOMP_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_omp_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TOMP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TGP_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tgp_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TGP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TACGP_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tacgp_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TACGP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TMP_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tmp_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TMP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TOOLS_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tools_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TOOLS_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TNCGMP_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tncgmp_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TNCGMP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TAFS_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tafs_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TAFS_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TCDLASSO_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tcdlasso_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TCDLASSO_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TCDENET_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tcdenet_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TCDENET_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TCDIENET_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param groups 1-based group ids, one per predictor
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tcdienet_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    IntegerVector groups,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TCDIENET_Solver(h->X_map_, h->D_map_, h->y_map_, groups_from_r(groups), opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

//' @title Create TCDTIKNET_Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param tikhonov_K p x p PSD Tikhonov matrix (dgCMatrix or dense)
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tcdtiknet_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    SEXP tikhonov_K,
    Rcpp::List options
) {
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(map_of(X), map_of(D), map_of(y));
    h->solver_ = new TCDTIKNET_Solver(h->X_map_, h->D_map_, h->y_map_, tsolver_as_sparse_matrix(tikhonov_K), opts);
    // The protection slot pins X, D and y until the finalizer has run (a
    // writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X, D, y));
}

// ==============================================================================
// Instance creators (memory-mapped X)
// ==============================================================================

//' @title Create TLARS_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_lars_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TLARS_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TLASSO_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tlasso_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TLASSO_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TSTEPWISE_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tstepwise_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TSTEPWISE_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TSTAGEWISE_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tstagewise_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TSTAGEWISE_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TENET_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tenet_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TENET_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TENETAug_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tenet_aug_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TENETAug_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TIENETAug_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param groups 1-based group ids, one per predictor
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tienet_aug_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    IntegerVector groups,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TIENETAug_Solver(h->X_map_, h->D_map_, h->y_map_, groups_from_r(groups), opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TIENET_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param groups 1-based group ids, one per predictor
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tienet_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    IntegerVector groups,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TIENET_Solver(h->X_map_, h->D_map_, h->y_map_, groups_from_r(groups), opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TIKNET_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param tikhonov_K p x p PSD Tikhonov matrix (dgCMatrix or dense)
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tiknet_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    SEXP tikhonov_K,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TIKNET_Solver(h->X_map_, h->D_map_, h->y_map_, tsolver_as_sparse_matrix(tikhonov_K), opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TOMP_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_omp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TOMP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TGP_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tgp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TGP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TACGP_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tacgp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TACGP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TMP_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tmp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TMP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TOOLS_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tools_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TOOLS_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TNCGMP_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tncgmp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TNCGMP_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}


//' @title Create TAFS_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tafs_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TAFS_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TCDLASSO_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tcdlasso_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TCDLASSO_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TCDENET_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tcdenet_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TCDENET_Solver(h->X_map_, h->D_map_, h->y_map_, opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TCDIENET_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param groups 1-based group ids, one per predictor
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tcdienet_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    IntegerVector groups,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TCDIENET_Solver(h->X_map_, h->D_map_, h->y_map_, groups_from_r(groups), opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

//' @title Create TCDTIKNET_Solver (memory-mapped X)
//'
//' @param X_ptr XPtr to a MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//' @param tikhonov_K p x p PSD Tikhonov matrix (dgCMatrix or dense)
//' @param options List from tsolver_options()
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tcdtiknet_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Rcpp::NumericMatrix D,
    Rcpp::NumericVector y,
    SEXP tikhonov_K,
    Rcpp::List options
) {
    auto X_map = X_ptr->getMap();
    const SolverOptions opts = parse_solver_options(options);
    auto h = std::make_unique<TSolverRcpp>(X_map, map_of(D), map_of(y));
    h->solver_ = new TCDTIKNET_Solver(h->X_map_, h->D_map_, h->y_map_, tsolver_as_sparse_matrix(tikhonov_K), opts);
    // The protection slot pins the mmap holder, D and y until the finalizer
    // has run (a writer's restore() writes back through the maps).
    return XPtr<TSolverRcpp>(h.release(), true, R_NilValue,
                             Rcpp::List::create(X_ptr, D, y));
}

// ==============================================================================
// Common TSolver API endpoints (called via XPtr<TSolverRcpp>)
// ==============================================================================

//' @title Execute Step in TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @param T_stop Number of steps to compute
//' @param early_stop Whether to stop early based on constraints
//' @noRd
// [[Rcpp::export]]
void tsolver_execute_step(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    int T_stop,
    bool early_stop = true
) {
    // Treat -1 or any negative as full computation (max size_t)
    std::size_t c_T_stop = (T_stop < 0) ? -1 : static_cast<std::size_t>(T_stop);
    ptr->solver_->executeStep(c_T_stop, early_stop);
}


//' @title Get Beta from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @param step Step to retrieve beta for
//'
//' @return Beta vector
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd tsolver_get_beta(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    int step = -1
) {
    return ptr->solver_->getBeta(step);
}


//' @title Get Cp scores from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of Cp scores
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_cp(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    auto cp = ptr->solver_->getCp();
    return Rcpp::wrap(cp);
}


//' @title Get RSS from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of RSS scores
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_rss(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getRSS());
}


//' @title Get R2 from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of R2 scores
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_r2(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getR2());
}


//' @title Get Degrees of Freedom from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of DoF
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_dof(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getDoF());
}


//' @title Get Active Set from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of active variable indices
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_actives(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getActives());
}


//' @title Get Inactive Set from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of inactive variable indices
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_inactives(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getInactives());
}


//' @title Get Intercept from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @param step Step to retrieve the intercept for
//' @return Numeric intercept value
//' @noRd
// [[Rcpp::export]]
double tsolver_get_intercept(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    int step = -1
) {
    return ptr->solver_->getIntercept(step);
}


//' @noRd
// [[Rcpp::export]]
int tsolver_get_num_steps(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->solver_->getNumSteps());
}


//' @noRd
// [[Rcpp::export]]
double tsolver_get_memory_bytes(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    // double: a byte count overflows R's 32-bit integers at 2 GiB
    return static_cast<double>(ptr->solver_->memoryBytes());
}


//' @noRd
// [[Rcpp::export]]
List tsolver_get_actions(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    const auto& actions = ptr->solver_->getActions();
    List result(actions.size());
    for (std::size_t i = 0; i < actions.size(); ++i)
        result[i] = Rcpp::wrap(actions[i]);
    return result;
}


// ==============================================================================
// Solver-Specific API Endpoints
// ==============================================================================

//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_lambda(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    if (auto* p = dynamic_cast<TLARS_Solver*>(ptr->solver_)) {
        return Rcpp::wrap(p->getLambda());
    }
    stop("tsolver_get_lambda: Solver type does not have a lambda path.");
}


//' @noRd
// [[Rcpp::export]]
int tsolver_get_num_removals(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    if (auto* p = dynamic_cast<TSTAGEWISE_Solver*>(ptr->solver_)) {
        return static_cast<int>(p->getNumRemovals());
    }
    if (auto* p = dynamic_cast<TENET_Solver*>(ptr->solver_)) {
        return static_cast<int>(p->getNumRemovals());
    }
    if (auto* p = dynamic_cast<TLASSO_Solver*>(ptr->solver_)) {
        return static_cast<int>(p->getNumRemovals());
    }
    stop("tsolver_get_num_removals: Solver type does not track removals.");
}


//' @noRd
// [[Rcpp::export]]
double tsolver_get_cycling_ratio(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    if (auto* p = dynamic_cast<TSTAGEWISE_Solver*>(ptr->solver_)) {
        return p->getCyclingRatio();
    }
    if (auto* p = dynamic_cast<TENET_Solver*>(ptr->solver_)) {
        return p->getCyclingRatio();
    }
    if (auto* p = dynamic_cast<TLASSO_Solver*>(ptr->solver_)) {
        return p->getCyclingRatio();
    }
    stop("tsolver_get_cycling_ratio: Solver type does not support getCyclingRatio().");
}


//' @noRd
// [[Rcpp::export]]
double tsolver_tienet_aug_get_lambda2(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    if (auto* p = dynamic_cast<TIENETAug_Solver*>(ptr->solver_)) {
        return p->getLambda2();
    }
    stop("tsolver_tienet_aug_get_lambda2: Solver is not a TIENETAug_Solver.");
}


//' @noRd
// [[Rcpp::export]]
IntegerVector tsolver_tienet_aug_get_groups(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    if (auto* p = dynamic_cast<TIENETAug_Solver*>(ptr->solver_)) {
        // Solver stores 0-based group ids; return 1-based (package convention).
        const Eigen::VectorXi& g = p->getGroups();
        IntegerVector out(g.size());
        for (Eigen::Index i = 0; i < g.size(); ++i) out[i] = g[i] + 1;
        return out;
    }
    stop("tsolver_tienet_aug_get_groups: Solver is not a TIENETAug_Solver.");
}


//' @noRd
// [[Rcpp::export]]
int tsolver_tienet_aug_get_num_groups(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    if (auto* p = dynamic_cast<TIENETAug_Solver*>(ptr->solver_)) {
        return static_cast<int>(p->getNumGroups());
    }
    stop("tsolver_tienet_aug_get_num_groups: Solver is not a TIENETAug_Solver.");
}


//' @title Save TSolver to File
//'
//' @param ptr TSolverRcpp holder pointer
//' @param filename Output filename
//' @noRd
// [[Rcpp::export]]
void tsolver_save(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        stop("tsolver_save: Cannot open file for saving.");
    }
    cereal::PortableBinaryOutputArchive archive(os);

    TSolver_Base* s = ptr->solver_;
    // Derived classes must be checked before their base classes.
    // LARS family
    if (auto* p = dynamic_cast<TSTAGEWISE_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TENETAug_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TIENETAug_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TIENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TIKNET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TLASSO_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TSTEPWISE_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TLARS_Solver*>(s)) {
        archive(*p);
    // OMP family
    } else if (auto* p = dynamic_cast<TACGP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TGP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TMP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TOOLS_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TOMP_Solver*>(s)) {
        archive(*p);
    // CD family
    } else if (auto* p = dynamic_cast<TCDTIKNET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TCDIENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TCDENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TCDLASSO_Solver*>(s)) {
        archive(*p);
    // Independent families
    } else if (auto* p = dynamic_cast<TNCGMP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TAFS_Solver*>(s)) {
        archive(*p);
    } else {
        stop("tsolver_save: Unsupported TSolver type.");
    }
}


//' @title Restore the caller's X and D and release the solver's claim
//'
//' @param ptr XPtr to TSolverRcpp
//'
//' @details Undoes the in-place centering/scaling a writer solver
//'   (normalize or intercept TRUE) applied to X and D at construction and
//'   releases its claim on the buffers, so that another writer may take
//'   them; y is never modified. No-op for a reader.
//' @noRd
// [[Rcpp::export]]
void tsolver_restore(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    ptr->solver_->restore(ptr->X_map_, ptr->D_map_, ptr->y_map_);
}

//' @title Load TSolver from File
//'
//' @param ptr TSolverRcpp holder pointer
//' @param filename Input filename
//' @noRd
// [[Rcpp::export]]
void tsolver_load(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        stop("tsolver_load: Cannot open file for loading.");
    }
    cereal::PortableBinaryInputArchive archive(is);

    TSolver_Base* s = ptr->solver_;
    // Derived classes must be checked before their base classes.
    // LARS family
    if (auto* p = dynamic_cast<TSTAGEWISE_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TENETAug_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TIENETAug_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TIENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TIKNET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TLASSO_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TSTEPWISE_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TLARS_Solver*>(s)) {
        archive(*p);
    // OMP family
    } else if (auto* p = dynamic_cast<TACGP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TGP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TMP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TOOLS_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TOMP_Solver*>(s)) {
        archive(*p);
    // CD family
    } else if (auto* p = dynamic_cast<TCDTIKNET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TCDIENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TCDENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TCDLASSO_Solver*>(s)) {
        archive(*p);
    // Independent families
    } else if (auto* p = dynamic_cast<TNCGMP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TAFS_Solver*>(s)) {
        archive(*p);
    } else {
        stop("tsolver_load: Unsupported TSolver type.");
    }
    // X_ and D_ are not serialised; reconnect to the holder's stable maps.
    ptr->solver_->reconnect(ptr->X_map_, ptr->D_map_);
}


// ==============================================================================
// Family getters of TIENET, the CD family and the sparse-K solvers
// ==============================================================================

//' @title TIENET: group-ridge penalty used
//' @noRd
// [[Rcpp::export]]
double tsolver_tienet_get_lambda2(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TIENET_Solver*>(ptr->solver_)) return p->getLambda2();
    stop("tsolver_tienet_get_lambda2: Solver is not a TIENET_Solver.");
}

//' @title TIENET: 1-based group ids, one per predictor
//' @noRd
// [[Rcpp::export]]
IntegerVector tsolver_tienet_get_groups(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TIENET_Solver*>(ptr->solver_)) {
        const Eigen::VectorXi& g = p->getGroups();
        IntegerVector out(g.size());
        for (Eigen::Index i = 0; i < g.size(); ++i) out[i] = g[i] + 1;
        return out;
    }
    stop("tsolver_tienet_get_groups: Solver is not a TIENET_Solver.");
}

//' @title TIENET: number of groups
//' @noRd
// [[Rcpp::export]]
int tsolver_tienet_get_num_groups(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TIENET_Solver*>(ptr->solver_)) return static_cast<int>(p->getNumGroups());
    stop("tsolver_tienet_get_num_groups: Solver is not a TIENET_Solver.");
}

//' @title TCDIENET: 1-based group ids, one per predictor
//' @noRd
// [[Rcpp::export]]
IntegerVector tsolver_tcdienet_get_groups(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TCDIENET_Solver*>(ptr->solver_)) {
        // group_of_ covers the unified (p + L) columns; the first p entries
        // are the original variables (dummy layers copy their variable's group).
        const std::vector<int>& g = p->getGroupOf();
        const auto np = static_cast<int>(p->getDummyStartIndex());
        IntegerVector out(np);
        for (int i = 0; i < np; ++i) out[i] = g[static_cast<std::size_t>(i)] + 1;
        return out;
    }
    stop("tsolver_tcdienet_get_groups: Solver is not a TCDIENET_Solver.");
}

//' @title TCDIENET: number of groups
//' @noRd
// [[Rcpp::export]]
int tsolver_tcdienet_get_num_groups(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TCDIENET_Solver*>(ptr->solver_)) return static_cast<int>(p->getNumGroups());
    stop("tsolver_tcdienet_get_num_groups: Solver is not a TCDIENET_Solver.");
}

//' @title CD family: lambda_1 at every recorded step
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_cd_get_lambda(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TCDLASSO_Solver*>(ptr->solver_)) return wrap(p->getLambda());
    stop("tsolver_cd_get_lambda: Solver is not a CD-family solver.");
}

//' @title CD family: quadratic penalty weight lambda_2 (0 for TCDLASSO)
//' @noRd
// [[Rcpp::export]]
double tsolver_cd_get_lambda2(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TCDLASSO_Solver*>(ptr->solver_)) return p->getLambda2();
    stop("tsolver_cd_get_lambda2: Solver is not a CD-family solver.");
}

//' @title CD family: certified crossing localization active?
//' @noRd
// [[Rcpp::export]]
bool tsolver_cd_is_certified_localization(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TCDLASSO_Solver*>(ptr->solver_)) return p->isCertifiedLocalization();
    stop("tsolver_cd_is_certified_localization: Solver is not a CD-family solver.");
}

//' @title CD family: did certification fall back to LEGACY along the path?
//' @noRd
// [[Rcpp::export]]
bool tsolver_cd_is_certified_degraded(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TCDLASSO_Solver*>(ptr->solver_)) return p->isCertifiedDegraded();
    stop("tsolver_cd_is_certified_degraded: Solver is not a CD-family solver.");
}

//' @title CD family: duality gap at every recorded step
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_cd_get_record_gaps(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TCDLASSO_Solver*>(ptr->solver_)) return wrap(p->getRecordGaps());
    stop("tsolver_cd_get_record_gaps: Solver is not a CD-family solver.");
}

//' @title CD family: KKT violation at every recorded step
//' @noRd
// NOLINTBEGIN(performance-unnecessary-value-param)
// [[Rcpp::export]]
NumericVector tsolver_cd_get_record_kkt_violations(XPtr<TSolverRcpp> ptr) {
    if (auto* p = dynamic_cast<TCDLASSO_Solver*>(ptr->solver_)) return wrap(p->getRecordKktViolations());
    stop("tsolver_cd_get_record_kkt_violations: Solver is not a CD-family solver.");
}
// NOLINTEND(performance-unnecessary-value-param)

//' @title Sparse-K solvers: quadratic penalty weight lambda_2
//' @noRd
// [[Rcpp::export]]
double tsolver_tik_get_lambda2(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TIKNET_Solver*>(ptr->solver_))    return p->getLambda2();
    if (auto* p = dynamic_cast<TCDTIKNET_Solver*>(ptr->solver_)) return p->getLambda2();
    stop("tsolver_tik_get_lambda2: Solver is not a sparse-K solver.");
}

//' @title TCDTIKNET: FOLDED dummy coupling active?
//' @noRd
// [[Rcpp::export]]
bool tsolver_tcdtiknet_is_folded(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    if (auto* p = dynamic_cast<TCDTIKNET_Solver*>(ptr->solver_)) return p->isFoldedDummyCoupling();
    stop("tsolver_tcdtiknet_is_folded: Solver is not a TCDTIKNET_Solver.");
}


// ==============================================================================
// The base methods shared with the Python binding, plus the family getters.
// ==============================================================================

//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd tsolver_get_residuals(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    return ptr->solver_->getResiduals();
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd tsolver_get_beta_path(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    return ptr->solver_->getBetaPathSparse().dense();
}

//' @noRd
// [[Rcpp::export]]
std::vector<double> tsolver_get_cp_with_sigma(XPtr<TSolverRcpp> ptr,
    double sigma_hat_sq) { // NOLINT(performance-unnecessary-value-param)
    return ptr->solver_->getCp(sigma_hat_sq);
}

//' @noRd
// [[Rcpp::export]]
std::string tsolver_solver_type_to_string(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    return ptr->solver_->solverTypeToString();
}

//' @noRd
// NOLINTBEGIN(performance-unnecessary-value-param)
// [[Rcpp::export]]
std::vector<std::size_t> tsolver_get_dropped_indices(XPtr<TSolverRcpp> ptr) {
    return ptr->solver_->getDroppedIndices();
}
// NOLINTEND(performance-unnecessary-value-param)

//' @noRd
// NOLINTBEGIN(performance-unnecessary-value-param)
// [[Rcpp::export]]
std::vector<std::size_t> tsolver_get_active_predictor_indices(XPtr<TSolverRcpp> ptr) {
    return ptr->solver_->getActivePredictorIndices();
}
// NOLINTEND(performance-unnecessary-value-param)

//' @noRd
// NOLINTBEGIN(performance-unnecessary-value-param)
// [[Rcpp::export]]
std::vector<std::size_t> tsolver_get_active_dummy_indices(XPtr<TSolverRcpp> ptr) {
    return ptr->solver_->getActiveDummyIndices();
}
// NOLINTEND(performance-unnecessary-value-param)

//' @noRd
// [[Rcpp::export]]
int tsolver_get_num_actives(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    return static_cast<int>(ptr->solver_->getNumActives());
}

//' @noRd
// [[Rcpp::export]]
int tsolver_get_dummy_start_index(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    return static_cast<int>(ptr->solver_->getDummyStartIndex());
}

//' @noRd
// [[Rcpp::export]]
bool tsolver_is_connected(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    return ptr->solver_->isConnected();
}

//' @noRd
// [[Rcpp::export]]
void tsolver_set_tolerance(XPtr<TSolverRcpp> ptr, double tol) { // NOLINT(performance-unnecessary-value-param)
    ptr->solver_->setTolerance(tol);
}

//' @noRd
// [[Rcpp::export]]
void tsolver_set_tie_seed(XPtr<TSolverRcpp> ptr, double seed) { // NOLINT(performance-unnecessary-value-param)
    ptr->solver_->setTieSeed(static_cast<uint32_t>(seed));
}

//' @noRd
// [[Rcpp::export]]
void tsolver_set_num_threads(XPtr<TSolverRcpp> ptr, int n_threads) { // NOLINT(performance-unnecessary-value-param)
    ptr->solver_->setNumThreads(n_threads);
}

//' @noRd
// [[Rcpp::export]]
int tsolver_get_num_threads(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    return ptr->solver_->getNumThreads();
}

//' @noRd
// [[Rcpp::export]]
void tsolver_set_exchangeable_tie(XPtr<TSolverRcpp> ptr, double alpha,
    double floor) { // NOLINT(performance-unnecessary-value-param)
    ptr->solver_->setExchangeableTie(alpha, floor);
}

//' @noRd
// [[Rcpp::export]]
std::vector<std::string> tsolver_get_warnings(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    return ptr->solver_->getWarnings();
}

//' @noRd
// [[Rcpp::export]]
void tsolver_clear_warnings(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    ptr->solver_->clearWarnings();
}

// ---- family getters ----------------------------------------------------------

//' @noRd
// [[Rcpp::export]]
int tsolver_lars_get_kahan_refresh_interval(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    auto* s = dynamic_cast<TLARS_Solver*>(ptr->solver_);
    if (s == nullptr) stop("get_kahan_refresh_interval: not a LARS-family solver.");
    return static_cast<int>(s->getKahanRefreshInterval());
}

//' @noRd
// [[Rcpp::export]]
void tsolver_lars_set_kahan_refresh_interval(XPtr<TSolverRcpp> ptr,
    int interval) { // NOLINT(performance-unnecessary-value-param)
    auto* s = dynamic_cast<TLARS_Solver*>(ptr->solver_);
    if (s == nullptr) stop("set_kahan_refresh_interval: not a LARS-family solver.");
    s->setKahanRefreshInterval(static_cast<std::size_t>(interval));
}

//' @noRd
// [[Rcpp::export]]
double tsolver_tenet_get_alpha(XPtr<TSolverRcpp> ptr) { // NOLINT(performance-unnecessary-value-param)
    auto* s = dynamic_cast<TENET_Solver*>(ptr->solver_);
    if (s == nullptr) stop("get_alpha: not a TENET solver.");
    return s->getAlpha();
}

//' @noRd
// NOLINTBEGIN(performance-unnecessary-value-param)
// [[Rcpp::export]]
Eigen::SparseMatrix<double> tsolver_tik_get_tikhonov_k(XPtr<TSolverRcpp> ptr) {
    if (auto* a = dynamic_cast<TIKNET_Solver*>(ptr->solver_)) return a->getTikhonovK();
    stop("get_tikhonov_k: not a TIKNET solver.");
}
// NOLINTEND(performance-unnecessary-value-param)

//' @noRd
// [[Rcpp::export]]
double tsolver_cd_get_kkt_violation(XPtr<TSolverRcpp> ptr,
    double lambda1) { // NOLINT(performance-unnecessary-value-param)
    auto* s = dynamic_cast<TCDLASSO_Solver*>(ptr->solver_);
    if (s == nullptr) stop("get_kkt_violation: not a CD-family solver.");
    return s->getKktViolation(lambda1);
}
