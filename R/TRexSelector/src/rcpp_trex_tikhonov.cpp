// =========================================================================================
// rcpp_trex_tikhonov.cpp - Rcpp bindings for the TRexTikhonovSelector class.
// =========================================================================================

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_tikhonov.h"

// std includes
#include <string>
#include <utility>
#include <vector>

// project includes
#include <trex_selector_methods/ad/trex_tikhonov/tik_lambda2_power_rule.hpp>
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

    if (control.containsElementNamed("penalty_structure")) {
        std::string ps = control["penalty_structure"];
        if (ps == "SPARSE_GENERAL") {
            params.penalty_structure = TikPenaltyStructure::SPARSE_GENERAL;
        } else if (ps == "GROUP_CENTERING") {
            params.penalty_structure = TikPenaltyStructure::GROUP_CENTERING;
        } else {
            Rcpp::stop("Unknown penalty_structure: '" + ps +
                       "'. Use 'SPARSE_GENERAL' or 'GROUP_CENTERING'.");
        }
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
        } else if (method_str == "PILOT_KNEE") {
            params.lambda2_method = TikLambda2Method::PILOT_KNEE;
        } else {
            Rcpp::stop("Unknown lambda2_method: '" + method_str +
                       "'. Use 'PILOT_KNEE', 'CV_1SE_IEN_CCD', "
                       "'CV_MIN_IEN_CCD', 'CV_1SE_TIK_SVD' or "
                       "'CV_MIN_TIK_SVD'.");
        }
    }

    if (control.containsElementNamed("cv_n_folds")) {
        params.cv_n_folds = Rcpp::as<int>(control["cv_n_folds"]);
    }
    if (control.containsElementNamed("cv_n_lambda")) {
        params.cv_n_lambda =
            static_cast<Eigen::Index>(Rcpp::as<int>(control["cv_n_lambda"]));
    }
    if (control.containsElementNamed("cv_n_lambda1")) {
        params.cv_n_lambda1 =
            static_cast<Eigen::Index>(Rcpp::as<int>(control["cv_n_lambda1"]));
    }
    if (control.containsElementNamed("cv_n_lambda2")) {
        params.cv_n_lambda2 =
            static_cast<Eigen::Index>(Rcpp::as<int>(control["cv_n_lambda2"]));
    }
    if (control.containsElementNamed("cv_dfmax_ratio")) {
        params.cv_dfmax_ratio = Rcpp::as<double>(control["cv_dfmax_ratio"]);
    }
    if (control.containsElementNamed("cv_seed")) {
        params.cv_seed = Rcpp::as<int>(control["cv_seed"]);
    }

    // ---- Pilot rule (PILOT_KNEE) --------------------------------------------
    // Negative counts would wrap through size_t; refuse them here (the C++
    // constructor gates the rest: pilot_K >= 2, pilot_T >= 1, ...).
    auto as_count = [&](const char* name) -> std::size_t {
        const int v = Rcpp::as<int>(control[name]);
        if (v < 0) {
            Rcpp::stop(std::string(name) + " must be >= 0.");
        }
        return static_cast<std::size_t>(v);
    };
    if (control.containsElementNamed("pilot_grid") &&
        !Rf_isNull(control["pilot_grid"])) {
        params.pilot_grid = Rcpp::as<std::vector<double>>(control["pilot_grid"]);
    }
    if (control.containsElementNamed("pilot_K")) {
        params.pilot_K = as_count("pilot_K");
    }
    if (control.containsElementNamed("pilot_T")) {
        params.pilot_T = as_count("pilot_T");
    }
    if (control.containsElementNamed("pilot_L_multiplier")) {
        params.pilot_L_multiplier = as_count("pilot_L_multiplier");
    }
    if (control.containsElementNamed("pilot_knee_tolerance")) {
        params.pilot_knee_tolerance =
            Rcpp::as<double>(control["pilot_knee_tolerance"]);
    }
    if (control.containsElementNamed("pilot_statistic")) {
        std::string stat = control["pilot_statistic"];
        if (stat == "MAJORITY") {
            params.pilot_statistic = PilotStatistic::MAJORITY;
        } else if (stat == "CALIBRATED") {
            params.pilot_statistic = PilotStatistic::CALIBRATED;
        } else if (stat == "GROUP_MAJORITY") {
            params.pilot_statistic = PilotStatistic::GROUP_MAJORITY;
        } else {
            Rcpp::stop("Unknown pilot_statistic: '" + stat +
                       "'. Use 'MAJORITY', 'CALIBRATED', or 'GROUP_MAJORITY'.");
        }
    }
    if (control.containsElementNamed("pilot_group_fraction")) {
        params.pilot_group_fraction =
            Rcpp::as<double>(control["pilot_group_fraction"]);
    }
    if (control.containsElementNamed("pilot_shared_dummies")) {
        params.pilot_shared_dummies =
            Rcpp::as<bool>(control["pilot_shared_dummies"]);
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

// Shared tuner-diagnostics converter from rcpp_trex_gvs.cpp
extern Rcpp::List lambda2_cv_diagnostics_to_list(
    const trex::ml_methods::model_selection::lambda2_cv_diagnostics& d);


/** @brief A G x T count table (vector of per-grid-point rows) as an R matrix. */
template <typename T>
static Rcpp::NumericMatrix rows_to_matrix(const std::vector<std::vector<T>>& rows) {
    const int G = static_cast<int>(rows.size());
    const int Tn = (G > 0) ? static_cast<int>(rows[0].size()) : 0;
    Rcpp::NumericMatrix m(G, Tn);
    for (int g = 0; g < G; ++g) {
        for (int t = 0; t < Tn; ++t) {
            m(g, t) = static_cast<double>(rows[static_cast<std::size_t>(g)]
                                               [static_cast<std::size_t>(t)]);
        }
    }
    return m;
}


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


//' @title Get the lambda_2 tuner diagnostics
//' @noRd
// [[Rcpp::export]]
Rcpp::List trex_tikhonov_get_lambda2_cv_diagnostics(
    XPtr<RTRexTikhonovSelector> r_ptr) {
    return lambda2_cv_diagnostics_to_list(
        r_ptr->get()->getLambda2CVDiagnostics());
}


//' @title Get the lambda_2 pilot record (ran == FALSE when no pilot ran)
//' @noRd
// [[Rcpp::export]]
Rcpp::List trex_tikhonov_get_lambda2_pilot_result(
    XPtr<RTRexTikhonovSelector> r_ptr) {
    const Lambda2PilotResult& r = r_ptr->get()->getLambda2PilotResult();
    auto counts = [](const std::vector<std::size_t>& v) {
        Rcpp::NumericVector out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            out[static_cast<R_xlen_t>(i)] = static_cast<double>(v[i]);
        }
        return out;
    };
    return Rcpp::List::create(
        Rcpp::Named("ran")               = r.ran,
        Rcpp::Named("grid")              = Rcpp::wrap(r.grid),
        Rcpp::Named("counts")            = counts(r.counts),
        Rcpp::Named("curve_majority")    = rows_to_matrix(r.curve_majority),
        Rcpp::Named("curve_soft")        = rows_to_matrix(r.curve_soft),
        Rcpp::Named("calibrated")        = counts(r.calibrated),
        Rcpp::Named("group_majority")    = counts(r.group_majority),
        Rcpp::Named("groups_qualifying") = counts(r.groups_qualifying),
        Rcpp::Named("probe_seconds")     = Rcpp::wrap(r.probe_seconds),
        Rcpp::Named("index")             = static_cast<int>(r.index) + 1,  // 1-based
        Rcpp::Named("lambda2")           = r.lambda2,
        Rcpp::Named("max_selected")      = static_cast<double>(r.pick.max_selected),
        Rcpp::Named("n_ties")            = static_cast<double>(r.pick.n_ties),
        Rcpp::Named("at_lower_end")      = r.at_lower_end,
        Rcpp::Named("at_upper_end")      = r.at_upper_end,
        Rcpp::Named("K_pilot")           = static_cast<double>(r.K_pilot),
        Rcpp::Named("T_pilot")           = static_cast<double>(r.T_pilot),
        Rcpp::Named("L_0")               = static_cast<double>(r.L_0),
        Rcpp::Named("seconds")           = r.seconds
    );
}


//' @title The knee of the full lambda_2 profile: one selection per grid value
//' @noRd
// [[Rcpp::export]]
Rcpp::List trex_tikhonov_lambda2_power_rule_impl(
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector y,
    double tFDR,
    Rcpp::List tik_control_list,
    Rcpp::List trex_control_list,
    int seed,
    std::vector<double> grid,
    double knee_tolerance,
    bool verbose
) {
    // Every probe normalizes X in place and hands it back on return, so X
    // and y are read through their maps and never copied.
    Eigen::Map<Eigen::MatrixXd> X_map(X.begin(), X.nrow(), X.ncol());
    Eigen::Map<Eigen::VectorXd> y_map(y.begin(), y.size());
    const TRexTikhonovControlParameter tik_ctrl = merge_tikhonov_control(
        parse_tikhonov_parameter(tik_control_list),
        parse_control_parameter(trex_control_list));
    Lambda2PowerRuleConfig cfg;
    cfg.grid           = std::move(grid);
    cfg.knee_tolerance = knee_tolerance;
    const Lambda2PowerRuleResult r = tikhonov_lambda2_power_rule(
        X_map, y_map, tFDR, tik_ctrl, seed, cfg, verbose);

    auto one_based = [](const std::vector<std::size_t>& v) {
        Rcpp::IntegerVector out(static_cast<R_xlen_t>(v.size()));
        for (std::size_t i = 0; i < v.size(); ++i) {
            out[static_cast<R_xlen_t>(i)] = static_cast<int>(v[i]) + 1;
        }
        return out;
    };
    const auto G = static_cast<R_xlen_t>(r.probes.size());
    Rcpp::List          probes(G);
    Rcpp::NumericVector lambdas(G);
    Rcpp::NumericVector counts(G);
    for (R_xlen_t k = 0; k < G; ++k) {
        const Lambda2Probe& pr = r.probes[static_cast<std::size_t>(k)];
        probes[k] = Rcpp::List::create(
            Rcpp::Named("lambda2")  = pr.lambda2,
            Rcpp::Named("selected") = one_based(pr.selected),
            Rcpp::Named("T_stop")   = static_cast<double>(pr.T_stop),
            Rcpp::Named("seconds")  = pr.seconds);
        lambdas[k] = pr.lambda2;
        counts[k]  = static_cast<double>(pr.selected.size());
    }
    return Rcpp::List::create(
        Rcpp::Named("lambda2")      = r.lambda2,
        Rcpp::Named("index")        = static_cast<int>(r.index) + 1,  // 1-based
        Rcpp::Named("selected")     = one_based(r.selection()),
        Rcpp::Named("grid")         = lambdas,
        Rcpp::Named("counts")       = counts,
        Rcpp::Named("probes")       = probes,
        Rcpp::Named("max_selected") = static_cast<double>(r.pick.max_selected),
        Rcpp::Named("n_ties")       = static_cast<double>(r.pick.n_ties),
        Rcpp::Named("at_lower_end") = r.at_lower_end,
        Rcpp::Named("at_upper_end") = r.at_upper_end,
        Rcpp::Named("seconds")      = r.seconds
    );
}


//' @title Build K = Gamma^T Gamma from a Tikhonov operator Gamma
//' @noRd
// [[Rcpp::export]]
Eigen::SparseMatrix<double> trex_tikhonov_gamma_to_k(SEXP gamma) {
    return TRexTikhonovSelector::gammaToK(as_sparse_matrix(gamma));
}
