// =========================================================================================
// rcpp_trex_tikhonov.cpp - Rcpp bindings for the TRexTikhonovSelector class.
// =========================================================================================

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_tikhonov.h"

// std includes
#include <string>
#include <vector>

// project utils includes
#include <utils/memmap/memory_mapped_matrix.hpp>

// =========================================================================================

using namespace Rcpp;

// =========================================================================================

/**
 * @brief Convert an R matrix (dense numeric or dgCMatrix) to Eigen sparse.
 *
 * Accepting both spares the package a hard dependency on the Matrix
 * package: dense inputs are converted via sparseView().
 */
static Eigen::SparseMatrix<double> as_sparse_matrix(SEXP m) {
    if (Rf_isS4(m)) {
        return Rcpp::as<Eigen::SparseMatrix<double>>(m);
    }
    const Eigen::MatrixXd dense = Rcpp::as<Eigen::MatrixXd>(m);
    return dense.sparseView();
}


/**
 * @brief Parses an R list into a TRexTikhonovControlParameter struct.
 *
 * The nested trex_ctrl is merged later (in the wrapper constructor) from
 * the separately-supplied trex_control list.
 */
TRexTikhonovControlParameter parse_tikhonov_parameter(const Rcpp::List& control) {
    TRexTikhonovControlParameter params;

    if (control.containsElementNamed("tikhonov_K") &&
        !Rf_isNull(control["tikhonov_K"])) {
        params.tikhonov_K = as_sparse_matrix(control["tikhonov_K"]);
    }

    if (control.containsElementNamed("lambda_2")) {
        params.lambda_2 = control["lambda_2"];
    }

    if (control.containsElementNamed("lambda2_method")) {
        std::string method_str = control["lambda2_method"];
        if (method_str == "CV_1SE_IEN_CCD") {
            params.lambda2_method = TikLambda2Method::CV_1SE_IEN_CCD;
        } else if (method_str == "CV_MIN_IEN_CCD") {
            params.lambda2_method = TikLambda2Method::CV_MIN_IEN_CCD;
        } else if (method_str == "CV_1SE_TIK_SVD") {
            params.lambda2_method = TikLambda2Method::CV_1SE_TIK_SVD;
        } else if (method_str == "CV_MIN_TIK_SVD") {
            params.lambda2_method = TikLambda2Method::CV_MIN_TIK_SVD;
        } else {
            Rcpp::stop("Unknown lambda2_method: '" + method_str +
                       "'. Use 'CV_1SE_IEN_CCD', 'CV_MIN_IEN_CCD', "
                       "'CV_1SE_TIK_SVD', or 'CV_MIN_TIK_SVD'.");
        }
    }

    if (control.containsElementNamed("cv_n_folds")) {
        params.cv_n_folds = Rcpp::as<int>(control["cv_n_folds"]);
    }
    if (control.containsElementNamed("cv_n_lambda")) {
        params.cv_n_lambda =
            static_cast<Eigen::Index>(Rcpp::as<int>(control["cv_n_lambda"]));
    }
    if (control.containsElementNamed("cv_seed")) {
        params.cv_seed = Rcpp::as<int>(control["cv_seed"]);
    }

    if (control.containsElementNamed("fold_dummy_coupling")) {
        params.fold_dummy_coupling =
            Rcpp::as<bool>(control["fold_dummy_coupling"]);
    }
    if (control.containsElementNamed("cluster_dummies")) {
        params.cluster_dummies = Rcpp::as<bool>(control["cluster_dummies"]);
    }

    // 1-based R group IDs -> 0-based C++ cluster IDs (GVS convention).
    if (control.containsElementNamed("prior_groups") &&
        !Rf_isNull(control["prior_groups"])) {
        IntegerVector r_groups = control["prior_groups"];
        params.prior_groups = std::vector<Eigen::Index>(r_groups.size());
        for (int i = 0; i < r_groups.size(); ++i) {
            params.prior_groups[i] = r_groups[i] - 1;
        }
    }

    if (control.containsElementNamed("corr_max")) {
        params.corr_max = control["corr_max"];
    }

    if (control.containsElementNamed("hc_linkage")) {
        namespace hac = trex::ml_methods::clustering::hierarchical::agglomerative;
        std::string linkage = control["hc_linkage"];
        if (linkage == "Single") params.hc_linkage = hac::LinkageMethod::Single;
        else if (linkage == "Complete") params.hc_linkage = hac::LinkageMethod::Complete;
        else if (linkage == "Average") params.hc_linkage = hac::LinkageMethod::Average;
        else if (linkage == "WPGMA") params.hc_linkage = hac::LinkageMethod::WPGMA;
        else Rcpp::stop("Unsupported linkage method: " + linkage +
                        "\nSupported: Single, Complete, Average, WPGMA.");
    }

    return params;
}


// =========================================================================================
// Exports
// =========================================================================================

// Shared parser from rcpp_trex_selector.cpp
extern TRexControlParameter parse_control_parameter(const Rcpp::List& control);


//' @title Create TRexTikhonovSelector
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexTikhonovSelector> trex_tikhonov_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector y,
    double tFDR,
    Rcpp::List tik_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.begin(), X.nrow(), X.ncol());
    Eigen::Map<Eigen::VectorXd> y_map(y.begin(), y.size());
    TRexTikhonovControlParameter tik_control =
        parse_tikhonov_parameter(tik_control_list);
    TRexControlParameter trex_control =
        parse_control_parameter(trex_control_list);
    // Pin X and y until the finalizer has run: the base destructor writes the
    // de-normalized X back through the stored map (GC order is unspecified).
    return XPtr<RTRexTikhonovSelector>(
        new RTRexTikhonovSelector(X_map, y_map, tFDR, tik_control,
                                  trex_control, seed, verbose),
        true, R_NilValue, Rcpp::List::create(X, y));
}


//' @title Create TRexTikhonovSelector from MemoryMappedMatrix
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexTikhonovSelector> trex_tikhonov_mmap_create(
    XPtr<trex::utils::memmap::MemoryMappedMatrix<double>> X_ptr,
    Rcpp::NumericVector y,
    double tFDR,
    Rcpp::List tik_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    Eigen::Map<Eigen::VectorXd> y_map(y.begin(), y.size());
    TRexTikhonovControlParameter tik_control =
        parse_tikhonov_parameter(tik_control_list);
    TRexControlParameter trex_control =
        parse_control_parameter(trex_control_list);
    // Pin the mmap holder and y until the finalizer has run (see above).
    return XPtr<RTRexTikhonovSelector>(
        new RTRexTikhonovSelector(X_ptr->getMap(), y_map, tFDR, tik_control,
                                  trex_control, seed, verbose),
        true, R_NilValue, Rcpp::List::create(X_ptr, y));
}


//' @title Run TRexTikhonovSelector
//' @noRd
// [[Rcpp::export]]
void trex_tikhonov_select(XPtr<RTRexTikhonovSelector> r_ptr) {
    r_ptr->select();
}


//' @title Get the resolved lambda_2 (solver scale)
//' @noRd
// [[Rcpp::export]]
double trex_tikhonov_get_lambda2_used(XPtr<RTRexTikhonovSelector> r_ptr) {
    return r_ptr->get()->getLambda2Used();
}


//' @title Build K = Gamma^T Gamma from a Tikhonov operator Gamma
//' @noRd
// [[Rcpp::export]]
Eigen::SparseMatrix<double> trex_tikhonov_gamma_to_k(SEXP gamma) {
    return TRexTikhonovSelector::gammaToK(as_sparse_matrix(gamma));
}
