// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_da.h"

using namespace Rcpp;
using namespace trex::trex_selector_methods::trex_core;
using namespace trex::trex_selector_methods::trex_da;
using trx_ml_hac_LinkageMethod = trex::ml_methods::clustering::hierarchical::agglomerative::LinkageMethod;

// Expose the parse_control_parameter from rcpp_trex_selector.cpp
extern TRexControlParameter parse_control_parameter(const Rcpp::List& control);

/**
 * @brief Parses an R list into a TRexDAControlParameter C++ struct.
 * 
 * Extracts dependency-aware specific configurations like da_method (BT, AR1, EQUI), 
 * correlation coefficients, and threshold levels.
 * 
 * @param control List of parameters passed from R.
 * @return Populated TRexDAControlParameter instance.
 */
TRexDAControlParameter parse_da_parameter(const Rcpp::List& control) {
    TRexDAControlParameter params;

    if (control.containsElementNamed("da_method")) {
        std::string method = control["da_method"];
        if (method == "BT") params.method = DAMethod::BT;
        else if (method == "AR1") params.method = DAMethod::AR1;
        else if (method == "EQUI") params.method = DAMethod::EQUI;
        else Rcpp::stop("Unknown DAMethod: " + method);
    }

    if (control.containsElementNamed("cor_coef")) params.cor_coef = control["cor_coef"];
    if (control.containsElementNamed("rho_thr_DA")) params.rho_thr_DA = control["rho_thr_DA"];
    
    if (control.containsElementNamed("hc_linkage")) {
        std::string linkage = control["hc_linkage"];
        if (linkage == "Single") params.hc_linkage = trx_ml_hac_LinkageMethod::Single;
        else if (linkage == "Complete") params.hc_linkage = trx_ml_hac_LinkageMethod::Complete;
        else if (linkage == "Average") params.hc_linkage = trx_ml_hac_LinkageMethod::Average;
        else if (linkage == "WPGMA") params.hc_linkage = trx_ml_hac_LinkageMethod::WPGMA;
        else if (linkage == "Ward") params.hc_linkage = trx_ml_hac_LinkageMethod::Ward;
        else Rcpp::stop("Unknown LinkageMethod: " + linkage);
    }

    return params;
}

//' @title Create TRexDASelector
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexDASelector> trex_da_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector y,
    double tFDR,
    Rcpp::List da_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.begin(), X.nrow(), X.ncol());
    Eigen::Map<Eigen::VectorXd> y_map(y.begin(), y.size());
    TRexDAControlParameter da_control = parse_da_parameter(da_control_list);
    TRexControlParameter trex_control = parse_control_parameter(trex_control_list);
    // Pin X and y until the finalizer has run: the base destructor writes the
    // de-normalized X back through the stored map (GC order is unspecified).
    return XPtr<RTRexDASelector>(new RTRexDASelector(X_map, y_map, tFDR, da_control, trex_control, seed, verbose),
                                 true, R_NilValue, Rcpp::List::create(X, y));
}

//' @title Create TRexDASelector from mmap
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexDASelector> trex_da_mmap_create(
    XPtr<trex::utils::memmap::MemoryMappedMatrix<double>> X_ptr,
    Rcpp::NumericVector y,
    double tFDR,
    Rcpp::List da_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    Eigen::Map<Eigen::VectorXd> y_map(y.begin(), y.size());
    TRexDAControlParameter da_control = parse_da_parameter(da_control_list);
    TRexControlParameter trex_control = parse_control_parameter(trex_control_list);
    // Pin the mmap holder and y until the finalizer has run (see above).
    return XPtr<RTRexDASelector>(new RTRexDASelector(X_ptr->getMap(), y_map, tFDR, da_control, trex_control, seed, verbose),
                                 true, R_NilValue, Rcpp::List::create(X_ptr, y));
}

//' @title Run TRexDASelector
//' @noRd
// [[Rcpp::export]]
void trex_da_select(XPtr<RTRexDASelector> r_ptr) {
    r_ptr->select();
}

// =========================================================================================
// Getters
// =========================================================================================

//' @title Cast DA pointer to Base pointer
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexSelector> trex_da_to_base_ptr(XPtr<RTRexDASelector> r_ptr) {
    RTRexSelector* base_ptr = static_cast<RTRexSelector*>(r_ptr.get());
    return XPtr<RTRexSelector>(base_ptr, false);
}

//' @title Get DA Rho Threshold
//' @noRd
// [[Rcpp::export]]
double trex_da_get_rho_thresh(XPtr<RTRexDASelector> r_ptr) {
    return r_ptr->get()->getDAResult().rho_thresh;
}

//' @title Get DA Rho Grid
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd trex_da_get_rho_grid(XPtr<RTRexDASelector> r_ptr) {
    return r_ptr->get()->getDAResult().rho_grid;
}

//' @title Get DA Method String
//' @noRd
// [[Rcpp::export]]
std::string trex_da_get_method_string(XPtr<RTRexDASelector> r_ptr) {
    DAMethod m = r_ptr->get()->getDAResult().method;
    if (m == DAMethod::BT) return "BT";
    if (m == DAMethod::AR1) return "AR1";
    if (m == DAMethod::EQUI) return "EQUI";
    return "UNKNOWN";
}

//' @title Get DA Cor Coef
//' @noRd
// [[Rcpp::export]]
double trex_da_get_cor_coef(XPtr<RTRexDASelector> r_ptr) {
    return r_ptr->get()->getDAResult().cor_coef;
}

//' @title Get DA FDP Hat Array BT
//' @noRd
// [[Rcpp::export]]
std::vector<Eigen::MatrixXd> trex_da_get_fdp_hat_array_bt(XPtr<RTRexDASelector> r_ptr) {
    return r_ptr->get()->getDAResult().FDP_hat_array_BT;
}

//' @title Get DA Phi Array BT
//' @noRd
// [[Rcpp::export]]
std::vector<Eigen::MatrixXd> trex_da_get_phi_array_bt(XPtr<RTRexDASelector> r_ptr) {
    return r_ptr->get()->getDAResult().Phi_array_BT;
}

//' @title Get DA R Array BT
//' @noRd
// [[Rcpp::export]]
std::vector<Eigen::MatrixXd> trex_da_get_r_array_bt(XPtr<RTRexDASelector> r_ptr) {
    return r_ptr->get()->getDAResult().R_array_BT;
}
