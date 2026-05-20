// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_da_wrapper.h"

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
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::VectorXd> y,
    double tFDR,
    Rcpp::List da_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    TRexDAControlParameter da_control = parse_da_parameter(da_control_list);
    TRexControlParameter trex_control = parse_control_parameter(trex_control_list);
    return XPtr<RTRexDASelector>(new RTRexDASelector(X, y, tFDR, da_control, trex_control, seed, verbose));
}

//' @title Create TRexDASelector from mmap
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexDASelector> trex_da_mmap_create(
    XPtr<trex::utils::memmap::MemoryMappedMatrix<double>> X_ptr,
    Eigen::Map<Eigen::VectorXd> y,
    double tFDR,
    Rcpp::List da_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    TRexDAControlParameter da_control = parse_da_parameter(da_control_list);
    TRexControlParameter trex_control = parse_control_parameter(trex_control_list);
    return XPtr<RTRexDASelector>(new RTRexDASelector(X_ptr->getMap(), y, tFDR, da_control, trex_control, seed, verbose));
}

//' @title Run TRexDASelector
//' @noRd
// [[Rcpp::export]]
void trex_da_select(XPtr<RTRexDASelector> r_ptr) {
    r_ptr->select();
}

//' @title Get DA Results
//' @noRd
// [[Rcpp::export]]
List trex_da_get_results(XPtr<RTRexDASelector> r_ptr) {
    const auto& res = r_ptr->get()->getDAResult();
    
    // Map enums to strings for R
    std::string method_str = (res.method == DAMethod::BT) ? "BT" : 
                             (res.method == DAMethod::AR1) ? "AR1" : "EQUI";
                             
    return List::create(
        Named("selected_var") = res.selected_var,
        Named("v_thresh") = res.v_thresh,
        Named("T_stop") = res.T_stop,
        Named("num_dummies") = res.num_dummies,
        Named("dummy_factor_L") = res.dummy_factor_L,
        Named("voting_grid") = res.voting_grid,
        Named("rho_thresh") = res.rho_thresh,
        Named("rho_grid") = res.rho_grid,
        Named("method") = method_str,
        Named("cor_coef") = res.cor_coef
    );
}

//' @title Get DA Matrices
//' @noRd
// [[Rcpp::export]]
List trex_da_get_matrices(XPtr<RTRexDASelector> r_ptr) {
    const auto& res = r_ptr->get()->getDAResult();
    
    return List::create(
        Named("R_mat") = res.R_mat,
        Named("FDP_hat_mat") = res.FDP_hat_mat,
        Named("Phi_mat") = res.Phi_mat,
        Named("Phi_prime") = res.Phi_prime,
        Named("FDP_hat_array_BT") = res.FDP_hat_array_BT,
        Named("Phi_array_BT") = res.Phi_array_BT,
        Named("R_array_BT") = res.R_array_BT
    );
}

//' @title Get DA Selected Indices
//' @noRd
// [[Rcpp::export]]
IntegerVector trex_da_get_selected_indices(XPtr<RTRexDASelector> r_ptr) {
    auto indices = r_ptr->get()->getSelectedIndices();
    IntegerVector res(indices.size());
    for(size_t i = 0; i < indices.size(); ++i) {
        res[i] = static_cast<int>(indices[i] + 1); // 1-based indexing for R
    }
    return res;
}
