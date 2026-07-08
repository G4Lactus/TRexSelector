// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_biobank_screening.h"

using namespace Rcpp;

extern TRexControlParameter parse_control_parameter(const Rcpp::List& control);
extern ScreenTRexControlParameter parse_screen_parameter(const Rcpp::List& control);

BiobankScreenTRexControl parse_biobank_screen_parameter(
    const Rcpp::List& biobank_control,
    const Rcpp::List& screen_control,
    const Rcpp::List& trex_control
) {
    BiobankScreenTRexControl params;
    
    if (biobank_control.containsElementNamed("target_FDR_trex")) params.target_FDR_trex = biobank_control["target_FDR_trex"];
    if (biobank_control.containsElementNamed("lower_bound_FDR")) params.lower_bound_FDR = biobank_control["lower_bound_FDR"];
    if (biobank_control.containsElementNamed("upper_bound_FDR")) params.upper_bound_FDR = biobank_control["upper_bound_FDR"];
    
    // The Screen-TRex control nests the base T-Rex control (single source of
    // truth shared by the Screen-TRex path and the plain T-Rex fallback).
    params.trex_screen_ctrl = parse_screen_parameter(screen_control);
    params.trex_screen_ctrl.trex_ctrl = parse_control_parameter(trex_control);
    
    return params;
}

// ==============================================================================

//' @title Create RTRexBiobankScreeningSelector 1D
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexBiobankScreeningSelector> trex_biobank_screening_1d_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector y,
    Rcpp::List biobank_control_list,
    Rcpp::List screen_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.begin(), X.nrow(), X.ncol());
    Eigen::Map<Eigen::VectorXd> y_map(y.begin(), y.size());
    BiobankScreenTRexControl control = parse_biobank_screen_parameter(biobank_control_list, screen_control_list, trex_control_list);
    // Pin X and y until the finalizer has run: the base destructor writes the
    // de-normalized X back through the stored map (GC order is unspecified).
    return XPtr<RTRexBiobankScreeningSelector>(new RTRexBiobankScreeningSelector(X_map, y_map, control, seed, verbose),
                                               true, R_NilValue, Rcpp::List::create(X, y));
}

//' @title Create RTRexBiobankScreeningSelector 2D
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexBiobankScreeningSelector> trex_biobank_screening_2d_create(
    Rcpp::NumericMatrix X,
    Rcpp::NumericMatrix Y,
    Rcpp::List biobank_control_list,
    Rcpp::List screen_control_list,
    Rcpp::List trex_control_list,
    int seed,
    bool verbose
) {
    Eigen::Map<Eigen::MatrixXd> X_map(X.begin(), X.nrow(), X.ncol());
    Eigen::Map<Eigen::MatrixXd> Y_map(Y.begin(), Y.nrow(), Y.ncol());
    BiobankScreenTRexControl control = parse_biobank_screen_parameter(biobank_control_list, screen_control_list, trex_control_list);
    // Pin X and Y until the finalizer has run (see above).
    return XPtr<RTRexBiobankScreeningSelector>(new RTRexBiobankScreeningSelector(X_map, Y_map, control, seed, verbose),
                                               true, R_NilValue, Rcpp::List::create(X, Y));
}

// ==============================================================================

// Helper to convert std::vector<std::size_t> to 1-based R IntegerVector
IntegerVector convert_indices_1based(const std::vector<std::size_t>& cpp_indices) {
    IntegerVector r_indices(cpp_indices.size());
    for (std::size_t i = 0; i < cpp_indices.size(); ++i) {
        r_indices[i] = cpp_indices[i] + 1; // 1-based indexing for R
    }
    return r_indices;
}

// Build the canonical per-phenotype record (a 9-field named list). This is the
// single source of truth for the biobank result shape: the 1-D getter returns
// one such record, and the 2-D getter returns a plain list of them. Keeping the
// record identical across both paths mirrors the Python binding, which hands
// back one BiobankScreenTRexResult (1-D) or a list of them (matrix).
List biobank_result_to_record(const BiobankScreenTRexResult& res) {
    return List::create(
        Named("phenotype_index") = res.phenotype_index + 1, // 1-based for R
        Named("selected_indices") = convert_indices_1based(res.selected_indices),
        Named("estimated_FDR") = res.estimated_FDR,
        Named("method_used") = res.method_used,
        Named("estimated_FDR_screen_ordinary") = res.estimated_FDR_screen_ordinary,
        Named("estimated_FDR_screen_bootstrap") = res.estimated_FDR_screen_bootstrap,
        Named("selected_indices_screen_ordinary") = convert_indices_1based(res.selected_indices_screen_ordinary),
        Named("selected_indices_screen_bootstrap") = convert_indices_1based(res.selected_indices_screen_bootstrap),
        Named("used_fallback_trex") = res.used_fallback_trex
    );
}

//' @title Screen Single Phenotype
//' @noRd
// [[Rcpp::export]]
List trex_biobank_screening_screen_phenotype(XPtr<RTRexBiobankScreeningSelector> r_ptr) {
    return biobank_result_to_record(r_ptr->get()->screenPhenotype());
}

//' @title Screen Multiple Phenotypes
//' @noRd
// [[Rcpp::export]]
List trex_biobank_screening_screen_phenotypes(XPtr<RTRexBiobankScreeningSelector> r_ptr) {
    auto results = r_ptr->get()->screenPhenotypes();
    std::size_t n = results.size();

    // A plain list of per-phenotype records: res[[i]] has the same shape as the
    // 1-D return. The former struct-of-arrays layout ($statistics data.frame +
    // parallel index lists) is dropped; every field it carried now lives inside
    // each record, so no information is lost and the shape matches Python's
    // list[BiobankScreenTRexResult].
    List out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = biobank_result_to_record(results[i]);
    }
    return out;
}