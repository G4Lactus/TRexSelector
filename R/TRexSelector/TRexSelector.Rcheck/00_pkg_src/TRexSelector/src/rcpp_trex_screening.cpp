// =============================================================================
// rcpp_trex_screening.cpp - Rcpp bindings for Screen-TRexSelector
// =============================================================================

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_screening.h"

using namespace Rcpp;
using namespace trex::trex_selector_methods::trex_screening;
using namespace trex::trex_selector_methods::trex_core;

extern TRexControlParameter parse_control_parameter(const Rcpp::List& control);

/**
 * @brief Parses an R list into a ScreenTRexControlParameter C++ struct.
 *
 * Configures bootstrap iterations, confidence thresholds, and the base
 * T-Rex methodology mapped to the Screen-TRex variant.
 *
 * @param control List of parameters passed from R.
 * @return Populated ScreenTRexControlParameter instance.
 */
ScreenTRexControlParameter parse_screen_parameter(const Rcpp::List& control) {
    ScreenTRexControlParameter params;

    if (control.containsElementNamed("use_bootstrap_CI")) {
        params.use_bootstrap_CI = control["use_bootstrap_CI"];
    }
    if (control.containsElementNamed("R_boot")) {
        params.R_boot = control["R_boot"];
    }
    if (control.containsElementNamed("ci_grid_step")) {
        params.ci_grid_step = control["ci_grid_step"];
    }

    if (control.containsElementNamed("trex_method")) {
        std::string method = control["trex_method"];
        if (method == "TREX") {
            params.trex_method = ScreenTRexMethod::TREX;
        } else if (method == "TREX_DA_AR1") {
            params.trex_method = ScreenTRexMethod::TREX_DA_AR1;
        } else if (method == "TREX_DA_EQUI") {
            params.trex_method = ScreenTRexMethod::TREX_DA_EQUI;
        } else if (method == "TREX_DA_BLOCK_EQUI") {
            params.trex_method = ScreenTRexMethod::TREX_DA_BLOCK_EQUI;
        } else {
            Rcpp::stop("Unknown ScreenTRexMethod: " + method);
        }
    }

    return params;
}



//' @title Create RTRexScreeningSelector
//'
//' @details Instantiates a Screen-TRex selector object.
//'
//' @param X Design matrix (n x p).
//' @param y Response vector (n).
//' @param screen_control_list List of Screen-TRex specific control parameters.
//' @param trex_control_list List of general T-Rex control parameters.
//' @param seed Random seed for reproducibility.
//' @param verbose Whether to print progress messages.
//'
//' @return XPtr to RTRexScreeningSelector instance.
//'
//' @noRd
//'
// [[Rcpp::export]]
XPtr<RTRexScreeningSelector> trex_screening_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::VectorXd> y,
    Rcpp::List screen_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    ScreenTRexControlParameter screen_control = parse_screen_parameter(screen_control_list);
    TRexControlParameter trex_control = parse_control_parameter(trex_control_list);
    return XPtr<RTRexScreeningSelector>(new RTRexScreeningSelector(X, y, screen_control,
                                                                     trex_control, seed, verbose));
}



//' @title Create RTRexScreeningSelector from mmap
//'
//' @details Instantiates a Screen-TRex selector object using a memory-mapped design matrix
//'
//' @param X_ptr Pointer to MemoryMappedMatrix for the design matrix.
//' @param y Response vector (n).
//' @param screen_control_list List of Screen-TRex specific control parameters.
//' @param trex_control_list List of general T-Rex control parameters.
//' @param seed Random seed for reproducibility.
//' @param verbose Whether to print progress messages.
//'
//' @noRd
//'
// [[Rcpp::export]]
XPtr<RTRexScreeningSelector> trex_screening_mmap_create(
    XPtr<trex::utils::memmap::MemoryMappedMatrix<double>> X_ptr,
    Eigen::Map<Eigen::VectorXd> y,
    Rcpp::List screen_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    ScreenTRexControlParameter screen_control = parse_screen_parameter(screen_control_list);
    TRexControlParameter trex_control = parse_control_parameter(trex_control_list);
    return XPtr<RTRexScreeningSelector>(new RTRexScreeningSelector(X_ptr->getMap(),
                                        y, screen_control, trex_control, seed, verbose));
}



//' @title Run RTRexScreeningSelector
//'
//' @details Executes the selection procedure of the Screen-TRex selector.
//'
//' @param r_ptr Pointer to the RTRexScreeningSelector instance
//'
//' @noRd
//'
// [[Rcpp::export]]
void trex_screening_select(XPtr<RTRexScreeningSelector> r_ptr) {
    r_ptr->select();
}



//' @title Get estimated_FDR
//' @noRd
// [[Rcpp::export]]
double trex_screening_get_estimated_fdr(XPtr<RTRexScreeningSelector> r_ptr) {
    return r_ptr->get()->getScreenResult().estimated_FDR;
}

//' @title Get used_bootstrap
//' @noRd
// [[Rcpp::export]]
bool trex_screening_get_used_bootstrap(XPtr<RTRexScreeningSelector> r_ptr) {
    return r_ptr->get()->getScreenResult().used_bootstrap;
}

//' @title Get confidence_level
//' @noRd
// [[Rcpp::export]]
double trex_screening_get_confidence_level(XPtr<RTRexScreeningSelector> r_ptr) {
    return r_ptr->get()->getScreenResult().confidence_level;
}

//' @title Get estimated_correlation
//' @noRd
// [[Rcpp::export]]
double trex_screening_get_estimated_correlation(XPtr<RTRexScreeningSelector> r_ptr) {
    return r_ptr->get()->getScreenResult().estimated_correlation;
}

//' @title Get beta_mat
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd trex_screening_get_beta_mat(XPtr<RTRexScreeningSelector> r_ptr) {
    return r_ptr->get()->getScreenResult().beta_mat;
}

//' @title Get dummy_betas
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd trex_screening_get_dummy_betas(XPtr<RTRexScreeningSelector> r_ptr) {
    return r_ptr->get()->getScreenResult().dummy_betas;
}

//' @title Get Selected Indices
//'
//' @details Retrieves the indices of the selected variables from the Screen-TRex selector.
//'
//' @param r_ptr Pointer to the RTRexScreeningSelector instance.
//'
//' @noRd
//'
// [[Rcpp::export]]
IntegerVector trex_screening_get_selected_indices(XPtr<RTRexScreeningSelector> r_ptr) {
    const auto& indices = r_ptr->get()->getSelectedIndices();
    IntegerVector r_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        r_indices[i] = indices[i] + 1; // 1-based indexing for R
    }
    return r_indices;
}
