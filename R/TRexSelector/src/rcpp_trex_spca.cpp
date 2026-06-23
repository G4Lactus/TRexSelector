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
namespace tg = trex::trex_selector_methods::trex_gvs;

/*
 * Note on Roxygen Documentation:
 * All functions exported via Rcpp in this file use @noRd. They are internal
 * C++ bindings wrapped by the TRexSPCASelector R6 class.
 */

// =============================================================================

/**
 * @brief Parses an R list into a TRexSPCAControlParameter C++ struct.
 *
 * @param spca_ctrl_list R list with keys:
 *   mode ("ActiveSet"/"Thresholded"), lambda2_ridge_loadings,
 *   gvs_type ("EN"/"IEN"), lambda_2, lambda2_method,
 *   ("GCV"/"COND_NUM"/"CV_MIN"/"CV_1SE"), K, max_dummy_multiplier.
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

    if (spca_ctrl_list.containsElementNamed("lambda2_ridge_loadings")) {
        ctrl.lambda2_ridge_loadings = spca_ctrl_list["lambda2_ridge_loadings"];
    }

    // --- GVS sub-selector control ---
    if (spca_ctrl_list.containsElementNamed("gvs_type")) {
        std::string gvs_str = spca_ctrl_list["gvs_type"];
        if (gvs_str == "EN") {
            ctrl.gvs_ctrl.gvs_type = tg::GVSType::EN;
        } else if (gvs_str == "IEN") {
            ctrl.gvs_ctrl.gvs_type = tg::GVSType::IEN;
        } else {
            Rcpp::stop("Unknown GVSType: '" + gvs_str + "'. Use 'EN' or 'IEN'.");
        }
    }

    if (spca_ctrl_list.containsElementNamed("lambda_2")) {
        ctrl.gvs_ctrl.lambda_2 = spca_ctrl_list["lambda_2"];
    }

    if (spca_ctrl_list.containsElementNamed("lambda2_method")) {
        std::string method_str = spca_ctrl_list["lambda2_method"];
        if (method_str == "GCV") {
            ctrl.gvs_ctrl.lambda2_method = tg::LambdaSelectionMethod::GCV;
        } else if (method_str == "COND_NUM") {
            ctrl.gvs_ctrl.lambda2_method = tg::LambdaSelectionMethod::COND_NUM;
        } else if (method_str == "CV_MIN") {
            ctrl.gvs_ctrl.lambda2_method = tg::LambdaSelectionMethod::CV_MIN;
        } else if (method_str == "CV_1SE") {
            ctrl.gvs_ctrl.lambda2_method = tg::LambdaSelectionMethod::CV_1SE;
        } else {
            Rcpp::stop("Unknown lambda2_method: '" + method_str +
                       "'. Use 'GCV', 'COND_NUM', 'CV_MIN', or 'CV_1SE'.");
        }
    }

    // --- Base T-Rex sub-selector control ---
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
//' @description Orchestrates M rounds of ordinary PCA, T-Rex GVS FDR-controlled
//'   variable selection, and sparse loading assembly to produce sparse principal
//'   components.
//'
//' @param X Design matrix (n x p). Column-centered internally; restored on return.
//' @param M Integer, number of sparse PCs to extract.
//' @param tFDR Numeric, target false discovery rate in (0, 1).
//' @param spca_ctrl_list Named R list of control parameters (mode,
//'   lambda2_ridge_loadings, gvs_type, lambda_2, lambda2_method,
//'   K, max_dummy_multiplier).
//' @param seed Integer random seed. \code{-1} (default) for non-deterministic.
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
    Rcpp::List spca_ctrl_list,
    int seed
) {
    TRexSPCAControlParameter ctrl = parse_spca_control(spca_ctrl_list);

    TRexSPCA spca(X, static_cast<Eigen::Index>(M), tFDR, ctrl, seed);
    TRexSPCAResult res = spca.select();

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
