// =============================================================================
// rcpp_trex_spca.cpp - Rcpp bindings for TRexSPCA
// =============================================================================

// [[Rcpp::plugins(cpp20)]]
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_spca.h"

using namespace Rcpp;
using namespace trex::trex_selector_methods::trex_spca;

namespace tc = trex::trex_selector_methods::trex_core;

/*
 * Note on Roxygen Documentation:
 * All functions exported via Rcpp in this file use @noRd. They are internal
 * C++ bindings wrapped by the TRexSPCASelector R6 class.
 */

// =============================================================================

/**
 * @brief Parses an R list into a TRexSPCAControlParameter C++ struct.
 *
 * @param spca_ctrl_list R list with keys: mode ("ActiveSet"/"Thresholded"),
 *   lambda2, seed, K, max_dummy_multiplier.
 * @return Populated TRexSPCAControlParameter instance.
 */
TRexSPCAControlParameter parse_spca_control(const Rcpp::List& spca_ctrl_list) {
    TRexSPCAControlParameter ctrl;

    if (spca_ctrl_list.containsElementNamed("mode")) {
        std::string mode_str = spca_ctrl_list["mode"];
        if (mode_str == "ActiveSet") {
            ctrl.mode = SPCAMode::ActiveSet;
        } else if (mode_str == "Thresholded") {
            ctrl.mode = SPCAMode::Thresholded;
        } else {
            Rcpp::stop("Unknown SPCAMode: '" + mode_str +
                       "'. Use 'ActiveSet' or 'Thresholded'.");
        }
    }

    if (spca_ctrl_list.containsElementNamed("lambda2")) {
        ctrl.lambda2 = spca_ctrl_list["lambda2"];
    }
    if (spca_ctrl_list.containsElementNamed("seed")) {
        ctrl.seed = spca_ctrl_list["seed"];
    }
    if (spca_ctrl_list.containsElementNamed("K")) {
        ctrl.trex_ctrl.K = static_cast<std::size_t>(
            static_cast<int>(spca_ctrl_list["K"]));
    }
    if (spca_ctrl_list.containsElementNamed("max_dummy_multiplier")) {
        ctrl.trex_ctrl.max_dummy_multiplier = static_cast<std::size_t>(
            static_cast<int>(spca_ctrl_list["max_dummy_multiplier"]));
    }

    return ctrl;
}


//' @title Run T-Rex Sparse PCA selection
//'
//' @description Orchestrates M rounds of ordinary PCA, T-Rex FDR-controlled
//'   variable selection, and sparse loading assembly to produce sparse principal
//'   components.
//'
//' @param X Design matrix (n x p). Column-centered internally; restored on return.
//' @param M Integer, number of sparse PCs to extract.
//' @param tFDR Numeric, target false discovery rate in (0, 1).
//' @param spca_ctrl_list Named R list of control parameters (mode, lambda2, seed,
//'   K, max_dummy_multiplier).
//'
//' @return Named list:
//'   \itemize{
//'     \item \code{Z} Score matrix (n x M).
//'     \item \code{V} Loading matrix (p x M).
//'     \item \code{active_sets} List of M integer vectors (1-based indices).
//'     \item \code{adjusted_ev} Numeric vector of adjusted explained variance (M).
//'     \item \code{cumulative_ev} Numeric vector of cumulative explained variance (M).
//'   }
//' @noRd
// [[Rcpp::export]]
Rcpp::List trex_spca_select(
    Eigen::Map<Eigen::MatrixXd> X,
    int M,
    double tFDR,
    Rcpp::List spca_ctrl_list
) {
    TRexSPCAControlParameter ctrl = parse_spca_control(spca_ctrl_list);

    TRexSPCAResult res = TRexSPCA::select(X,
                                          static_cast<Eigen::Index>(M),
                                          tFDR,
                                          ctrl);

    // Convert active sets to R-friendly 1-based integer vectors
    Rcpp::List active_sets_r(static_cast<R_xlen_t>(M));
    for (int m = 0; m < M; ++m) {
        const auto& as_cpp = res.active_sets[static_cast<std::size_t>(m)];
        Rcpp::IntegerVector as_r(static_cast<R_xlen_t>(as_cpp.size()));
        for (std::size_t i = 0; i < as_cpp.size(); ++i) {
            as_r[static_cast<R_xlen_t>(i)] =
                static_cast<int>(as_cpp[i]) + 1; // 0-based → 1-based
        }
        active_sets_r[m] = as_r;
    }

    return Rcpp::List::create(
        Rcpp::Named("Z")            = res.Z,
        Rcpp::Named("V")            = res.V,
        Rcpp::Named("active_sets")  = active_sets_r,
        Rcpp::Named("adjusted_ev")  = res.adjusted_ev,
        Rcpp::Named("cumulative_ev")= res.cumulative_ev
    );
}
