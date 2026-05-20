// =========================================================================================
// rcpp_trex_selector.cpp - Rcpp bindings for core TRexSelector class.
// =========================================================================================

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_selector.h"

// std includes
#include <unordered_map>
#include <string>

// project trex includes
#include <trex_selector_methods/utils/trex_solver_dispatch.hpp>

// project utils includes
#include <utils/memmap/memory_mapped_matrix.hpp>

// =========================================================================================

// Namespace convenience
using namespace Rcpp;
using namespace trex::trex_selector_methods::trex_core;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;

// =========================================================================================

/**
 * @brief Parses an R list into a TRexControlParameter C++ struct.
 *
 * Maps general parameters for the TRex framework, such as the L-loop strategy,
 * solver type, parallel execution settings, and specific solver hyperparameters.
 *
 * @param control List of parameters passed from R.
 *
 * @return Populated TRexControlParameter instance.
 */
TRexControlParameter parse_control_parameter(const Rcpp::List& control) {
    TRexControlParameter params;

    // Experiment Configuration
    if (control.containsElementNamed("K")) {
        params.K = control["K"];
    }
    if (control.containsElementNamed("max_dummy_multiplier")) {
        params.max_dummy_multiplier = control["max_dummy_multiplier"];
    }
    if (control.containsElementNamed("use_max_T_stop")) {
        params.use_max_T_stop = control["use_max_T_stop"];
    }
    if (control.containsElementNamed("opt_threshold")) {
        params.opt_threshold = control["opt_threshold"];
    }

    // L-Loop Strategy
    if (control.containsElementNamed("lloop_strategy")) {
        std::string strategy = control["lloop_strategy"];
        if (strategy == "SKIPL") {
            params.lloop_strategy = LLoopStrategy::SKIPL;
        } else if (strategy == "STANDARD") {
            params.lloop_strategy = LLoopStrategy::STANDARD;
        } else if (strategy == "HCONCAT") {
            params.lloop_strategy = LLoopStrategy::HCONCAT;
        } else if (strategy == "PERMUTATION") {
            params.lloop_strategy = LLoopStrategy::PERMUTATION;
        } else if (strategy == "PERMUTATION_DIRECT") {
            params.lloop_strategy = LLoopStrategy::PERMUTATION_DIRECT;
        } else if (strategy == "DIRECT") {
            params.lloop_strategy = LLoopStrategy::DIRECT;
        } else {
            Rcpp::stop("Unknown LLoopStrategy: " + strategy);
        }
    }

    // T-Loop Early Stopping
    if (control.containsElementNamed("tloop_stagnation_stop")) {
        params.tloop_stagnation_stop = control["tloop_stagnation_stop"];
    }
    if (control.containsElementNamed("tloop_max_stagnant_steps")) {
        params.tloop_max_stagnant_steps = control["tloop_max_stagnant_steps"];
    }

    // Parallel configuration
    if (control.containsElementNamed("parallel_rnd_experiments")) {
        params.parallel_rnd_experiments = control["parallel_rnd_experiments"];
    }
    if (control.containsElementNamed("use_memory_mapping")) {
        params.use_memory_mapping = control["use_memory_mapping"];
    }

    // Solver Type mapping
    if (control.containsElementNamed("method")) {
        std::string method = control["method"];

        static const std::unordered_map<std::string, SolverTypeForTRex> solver_map = {
            {"TLARS",      SolverTypeForTRex::TLARS},
            {"TLASSO",     SolverTypeForTRex::TLASSO},
            {"TENET",      SolverTypeForTRex::TENET},
            {"TSTEPWISE",  SolverTypeForTRex::TSTEPWISE},
            {"TSTAGEWISE", SolverTypeForTRex::TSTAGEWISE},
            {"TOMP",       SolverTypeForTRex::TOMP},
            {"TGP",        SolverTypeForTRex::TGP},
            {"TACGP",      SolverTypeForTRex::TACGP},
            {"TMP",        SolverTypeForTRex::TMP},
            {"TNCGMP",     SolverTypeForTRex::TNCGMP},
            {"TOOLS",     SolverTypeForTRex::TOOLS},
            {"TAFS",      SolverTypeForTRex::TAFS}
        };

        auto it = solver_map.find(method);
        if (it != solver_map.end()) {
            params.solver_type = it->second;
        } else {
            Rcpp::stop("Unknown TRex selector method: " + method);
        }
    }

    // Hyperparameters
    if (control.containsElementNamed("lambda2")) {
        params.solver_params.lambda2 = control["lambda2"];
    }
    if (control.containsElementNamed("rho_afs")) {
        params.solver_params.rho_afs = control["rho_afs"];
    }
    if (control.containsElementNamed("ncgmp_variant")) {
        params.solver_params.ncgmp_variant = control["ncgmp_variant"];
    }
    if (control.containsElementNamed("tol")) {
        params.solver_params.tol = control["tol"];
    }

    params.autoConfigResources();
    return params;
}


//' @title Create TRexSelector
//'
//' @param X Design matrix (n × p).
//' @param y Response vector (n).
//' @param tFDR Target FDR level (default: 0.1).
//' @param control List of control parameters (see details).
//' @param seed Random seed (default: -1).
//' @param verbose Whether to print progress (default: TRUE).
//'
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexSelector> trex_selector_create(
    Eigen::Map<Eigen::MatrixXd> X, // NOLINT (performance-unnecessary-value-param)
    Eigen::Map<Eigen::VectorXd> y, // NOLINT (performance-unnecessary-value-param)
    double tFDR,
    const Rcpp::List& control_list,
    int seed,
    bool verbose
) {
    TRexControlParameter control = parse_control_parameter(control_list);
    return XPtr<RTRexSelector>(new RTRexSelector(X, y, tFDR,
                               control, seed, verbose));
}


//' @title Create TRexSelector from mmap
//'
//' @param X_ptr Pointer to MemoryMappedMatrix (n × p).
//' @param y Response vector (n).
//' @param tFDR Target FDR level (default: 0.1).
//' @param control List of control parameters (see details).
//' @param seed Random seed (default: -1).
//' @param verbose Whether to print progress (default: TRUE).
//'
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexSelector> trex_selector_mmap_create(
    XPtr<trex::utils::memmap::MemoryMappedMatrix<double>> X_ptr, // NOLINT (performance-unnecessary-value-param)
    Eigen::Map<Eigen::VectorXd> y, // NOLINT (performance-unnecessary-value-param)
    double tFDR,
    const Rcpp::List& control_list,
    int seed,
    bool verbose
) {
    TRexControlParameter control = parse_control_parameter(control_list);
    return XPtr<RTRexSelector>(new RTRexSelector(X_ptr->getMap(), y, tFDR,
                               control, seed, verbose));
}


//' @title Run TRexSelector
//'
//' @param r_ptr Pointer to RTRexSelector instance.
//'
//' @noRd
// [[Rcpp::export]]
void trex_selector_select(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    r_ptr->select();
}


// =========================================================================================
// Getters
// =========================================================================================

//' @title Get Selected Variables
//'
//' @param r_ptr Pointer to RTRexSelector instance.
//'
//' @noRd
// [[Rcpp::export]]
LogicalVector trex_selector_get_selected_var(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    auto var = r_ptr->getSelectedVar();
    LogicalVector res(static_cast<int>(var.size()));
    for (size_t i = 0; i < var.size(); ++i) {
        res[static_cast<int>(i)] = static_cast<bool>(var[i]);
    }
    return res;
}


//' @title Get Selected Indices
//'
//' @param r_ptr Pointer to RTRexSelector instance.
//'
//' @noRd
// [[Rcpp::export]]
IntegerVector trex_selector_get_selected_indices(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    auto indices = r_ptr->getSelectedIndices();
    IntegerVector res(static_cast<int>(indices.size()));
    for(size_t i = 0; i < indices.size(); ++i) {
        // 1-based indexing for R
        res[static_cast<int>(i)] = static_cast<int>(indices[i] + 1);
    }
    return res;
}


//' @title Get Voting Threshold
//' @noRd
// [[Rcpp::export]]
double trex_selector_get_voting_threshold(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return r_ptr->getVotingThreshold();
}


//' @title Get Stopping Time T
//' @noRd
// [[Rcpp::export]]
int trex_selector_get_t_stop(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return static_cast<int>(r_ptr->getTStop());
}


//' @title Get Dummy Multiplier L
//' @noRd
// [[Rcpp::export]]
int trex_selector_get_dummy_multiplier_l(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return static_cast<int>(r_ptr->getDummyMultiplierL());
}

//' @title Get Voting Grid
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd trex_selector_get_voting_grid(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return r_ptr->getVotingGrid();
}


//' @title Get FDP Hat Matrix
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd trex_selector_get_fdp_hat_mat(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return r_ptr->getFDPHatMat();
}


//' @title Get Phi Matrix
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd trex_selector_get_phi_mat(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return r_ptr->getPhiMat();
}


//' @title Get R Matrix
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd trex_selector_get_r_mat(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return r_ptr->getRMat();
}


//' @title Get Phi Prime
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd trex_selector_get_phi_prime(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return r_ptr->getPhiPrime();
}


//' @title Get X Means
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd trex_selector_get_x_means(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return r_ptr->getXMeans();
}


//' @title Get X L2 Norms
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd trex_selector_get_x_l2_norms(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return r_ptr->getXL2Norms();
}


//' @title Get y Mean
//' @noRd
// [[Rcpp::export]]
double trex_selector_get_y_mean(
    XPtr<RTRexSelector> r_ptr // NOLINT (performance-unnecessary-value-param)
) {
    return r_ptr->getYMean();
}
