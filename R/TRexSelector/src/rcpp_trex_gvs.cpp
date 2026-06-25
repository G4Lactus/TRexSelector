// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_gvs.h"

using namespace Rcpp;
using namespace trex::trex_selector_methods::trex_gvs;
using namespace trex::trex_selector_methods::trex_core;

// From rcpp_trex_selector.cpp
extern TRexControlParameter parse_control_parameter(const Rcpp::List& control);

/**
 * @brief Parses an R list into a TRexGVSControlParameter C++ struct.
 *
 * Maps parameters such as grouping correlation thresholds, linkage methods,
 * and lambda penalties associated with the Group Variable Selection variant.
 *
 * @param control List of parameters passed from R.
 * @return Populated TRexGVSControlParameter instance.
 */
TRexGVSControlParameter parse_gvs_parameter(const Rcpp::List& control) {
    TRexGVSControlParameter params;

    if (control.containsElementNamed("gvs_type")) {
        std::string method = control["gvs_type"];
        if (method == "EN") params.gvs_type = GVSType::EN;
        else if (method == "IEN") params.gvs_type = GVSType::IEN;
        else Rcpp::stop("Unknown GVSType: " + method);
    }

    if (control.containsElementNamed("corr_max")) params.corr_max = control["corr_max"];

    if (control.containsElementNamed("hc_linkage")) {
        std::string linkage = control["hc_linkage"];
        if (linkage == "Single") params.hc_linkage = trex::ml_methods::clustering::hierarchical::agglomerative::LinkageMethod::Single;
        else if (linkage == "Complete") params.hc_linkage = trex::ml_methods::clustering::hierarchical::agglomerative::LinkageMethod::Complete;
        else if (linkage == "Average") params.hc_linkage = trex::ml_methods::clustering::hierarchical::agglomerative::LinkageMethod::Average;
        else if (linkage == "WPGMA") params.hc_linkage = trex::ml_methods::clustering::hierarchical::agglomerative::LinkageMethod::WPGMA;
        else Rcpp::stop("Unsupported linkage method for GVS: " + linkage + "\nSupported: Single, Complete, Average, WPGMA.");
    }

    if (control.containsElementNamed("lambda_2")) params.lambda_2 = control["lambda_2"];

    if (control.containsElementNamed("lambda2_method")) {
        std::string method_str = control["lambda2_method"];
        if (method_str == "CV_1SE") {
            params.lambda2_method = LambdaSelectionMethod::CV_1SE;
        } else if (method_str == "CV_MIN") {
            params.lambda2_method = LambdaSelectionMethod::CV_MIN;
        } else if (method_str == "GCV") {
            params.lambda2_method = LambdaSelectionMethod::GCV;
        } else {
            Rcpp::stop("Unknown lambda2_method: '" + method_str +
                       "'. Use 'CV_1SE', 'CV_MIN', or 'GCV'.");
        }
    }

    if (control.containsElementNamed("groups") && !Rf_isNull(control["groups"])) {
        IntegerVector r_groups = control["groups"];
        params.prior_groups = std::vector<Eigen::Index>(r_groups.size());
        for(int i=0; i<r_groups.size(); ++i) params.prior_groups[i] = r_groups[i] - 1;
    }

    return params;
}

//' @title Create TRexGVSSelector
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexGVSSelector> trex_gvs_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::VectorXd> y,
    double tFDR,
    Rcpp::List gvs_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    TRexGVSControlParameter gvs_control = parse_gvs_parameter(gvs_control_list);
    TRexControlParameter trex_control = parse_control_parameter(trex_control_list);
    return XPtr<RTRexGVSSelector>(new RTRexGVSSelector(X, y, tFDR, gvs_control, trex_control, seed, verbose));
}

//' @title Create TRexGVSSelector from MemoryMappedMatrix
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexGVSSelector> trex_gvs_mmap_create(
    XPtr<trex::utils::memmap::MemoryMappedMatrix<double>> X_ptr,
    Eigen::Map<Eigen::VectorXd> y,
    double tFDR,
    Rcpp::List gvs_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    TRexGVSControlParameter gvs_control = parse_gvs_parameter(gvs_control_list);
    TRexControlParameter trex_control = parse_control_parameter(trex_control_list);
    return XPtr<RTRexGVSSelector>(new RTRexGVSSelector(X_ptr->getMap(), y, tFDR, gvs_control, trex_control, seed, verbose));
}

//' @title Run TRexGVSSelector
//' @noRd
// [[Rcpp::export]]
void trex_gvs_select(XPtr<RTRexGVSSelector> r_ptr) {
    r_ptr->select();
}

//' @title Get lambda2_used
//' @noRd
// [[Rcpp::export]]
double trex_gvs_get_lambda2_used(XPtr<RTRexGVSSelector> r_ptr) {
    return r_ptr->get()->getGVSResult().lambda2_used;
}

//' @title Get gvs_type
//' @noRd
// [[Rcpp::export]]
std::string trex_gvs_get_gvs_type(XPtr<RTRexGVSSelector> r_ptr) {
    switch(r_ptr->get()->getGVSResult().gvs_type) {
        case GVSType::EN: return "EN";
        case GVSType::IEN: return "IEN";
    }
    return "UNKNOWN";
}

//' @title Get max_clusters
//' @noRd
// [[Rcpp::export]]
int trex_gvs_get_max_clusters(XPtr<RTRexGVSSelector> r_ptr) {
    return static_cast<int>(r_ptr->get()->getGVSResult().max_clusters);
}

//' @title Get Selected Indices
//' @noRd
// [[Rcpp::export]]
IntegerVector trex_gvs_get_selected_indices(XPtr<RTRexGVSSelector> r_ptr) {
    const auto& indices = r_ptr->get()->getSelectedIndices();
    IntegerVector r_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        r_indices[i] = indices[i] + 1; // 1-based indexing for R
    }
    return r_indices;
}
