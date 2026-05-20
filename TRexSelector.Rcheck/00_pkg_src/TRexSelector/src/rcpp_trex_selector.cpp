// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "rcpp_trex_wrappers.h"
#include <trex_selector_methods/utils/trex_solver_dispatch.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>

using namespace Rcpp;
using namespace trex::trex_selector_methods::trex_core;
using trex::trex_selector_methods::utils::solver_dispatch::SolverTypeForTRex;

/**
 * @brief Parses an R list into a TRexControlParameter C++ struct.
 * 
 * Maps general parameters for the TRex framework, such as the L-loop strategy,
 * solver type, parallel execution settings, and specific solver hyperparameters.
 * 
 * @param control List of parameters passed from R.
 * @return Populated TRexControlParameter instance.
 */
TRexControlParameter parse_control_parameter(const Rcpp::List& control) {
    TRexControlParameter params;
    
    // Experiment Configuration
    if (control.containsElementNamed("K")) params.K = control["K"];
    if (control.containsElementNamed("max_dummy_multiplier")) params.max_dummy_multiplier = control["max_dummy_multiplier"];
    if (control.containsElementNamed("use_max_T_stop")) params.use_max_T_stop = control["use_max_T_stop"];
    if (control.containsElementNamed("opt_threshold")) params.opt_threshold = control["opt_threshold"];
    
    // L-Loop Strategy
    if (control.containsElementNamed("lloop_strategy")) {
        std::string strategy = control["lloop_strategy"];
        if (strategy == "SKIPL") params.lloop_strategy = LLoopStrategy::SKIPL;
        else if (strategy == "STANDARD") params.lloop_strategy = LLoopStrategy::STANDARD;
        else if (strategy == "HCONCAT") params.lloop_strategy = LLoopStrategy::HCONCAT;
        else if (strategy == "PERMUTATION") params.lloop_strategy = LLoopStrategy::PERMUTATION;
        else if (strategy == "PERMUTATION_DIRECT") params.lloop_strategy = LLoopStrategy::PERMUTATION_DIRECT;
        else if (strategy == "DIRECT") params.lloop_strategy = LLoopStrategy::DIRECT;
        else Rcpp::stop("Unknown LLoopStrategy: " + strategy);
    }

    // T-Loop Early Stopping
    if (control.containsElementNamed("tloop_stagnation_stop")) params.tloop_stagnation_stop = control["tloop_stagnation_stop"];
    if (control.containsElementNamed("tloop_max_stagnant_steps")) params.tloop_max_stagnant_steps = control["tloop_max_stagnant_steps"];

    // Parallel configuration
    if (control.containsElementNamed("parallel_rnd_experiments")) params.parallel_rnd_experiments = control["parallel_rnd_experiments"];
    if (control.containsElementNamed("use_memory_mapping")) params.use_memory_mapping = control["use_memory_mapping"];

    // Solver Type mapping
    if (control.containsElementNamed("method")) {
        std::string method = control["method"];
        if (method == "TLARS" || method == "LARS") params.solver_type = SolverTypeForTRex::TLARS;
        else if (method == "TLASSO" || method == "LASSO") params.solver_type = SolverTypeForTRex::TLASSO;
        else if (method == "TENET" || method == "ENET") params.solver_type = SolverTypeForTRex::TENET;
        else if (method == "TSTEPWISE" || method == "STEPWISE") params.solver_type = SolverTypeForTRex::TSTEPWISE;
        else if (method == "TSTAGEWISE" || method == "STAGEWISE") params.solver_type = SolverTypeForTRex::TSTAGEWISE;
        else if (method == "TOMP" || method == "OMP") params.solver_type = SolverTypeForTRex::TOMP;
        else if (method == "TGP" || method == "GP") params.solver_type = SolverTypeForTRex::TGP;
        else if (method == "TACGP" || method == "ACGP") params.solver_type = SolverTypeForTRex::TACGP;
        else if (method == "TMP" || method == "MP") params.solver_type = SolverTypeForTRex::TMP;
        else if (method == "TNCGMP" || method == "NCGMP") params.solver_type = SolverTypeForTRex::TNCGMP;
        else if (method == "TOOLS" || method == "OLS") params.solver_type = SolverTypeForTRex::TOOLS;
        else if (method == "TAFS" || method == "AFS") params.solver_type = SolverTypeForTRex::TAFS;
        else Rcpp::stop("Unknown TRex selector method: " + method);
    }
    
    // Hyperparameters
    if (control.containsElementNamed("lambda2")) params.solver_params.lambda2 = control["lambda2"];
    if (control.containsElementNamed("rho_afs")) params.solver_params.rho_afs = control["rho_afs"];
    if (control.containsElementNamed("ncgmp_variant")) params.solver_params.ncgmp_variant = control["ncgmp_variant"];

    params.autoConfigResources();
    return params;
}

//' @title Create TRexSelector
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexSelector> trex_selector_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::VectorXd> y,
    double tFDR,
    Rcpp::List control_list,
    int seed,
    bool verbose
) {
    TRexControlParameter control = parse_control_parameter(control_list);
    return XPtr<RTRexSelector>(new RTRexSelector(X, y, tFDR, control, seed, verbose));
}

//' @title Create TRexSelector from mmap
//' @noRd
// [[Rcpp::export]]
XPtr<RTRexSelector> trex_selector_mmap_create(
    XPtr<trex::utils::memmap::MemoryMappedMatrix<double>> X_ptr,
    Eigen::Map<Eigen::VectorXd> y,
    double tFDR,
    Rcpp::List control_list,
    int seed,
    bool verbose
) {
    TRexControlParameter control = parse_control_parameter(control_list);
    return XPtr<RTRexSelector>(new RTRexSelector(X_ptr->getMap(), y, tFDR, control, seed, verbose));
}

//' @title Run TRexSelector
//' @noRd
// [[Rcpp::export]]
List trex_selector_select(XPtr<RTRexSelector> r_ptr) {
    auto res = r_ptr->select();
    return List::create(
        Named("selected_var") = res.selected_var,
        Named("v_thresh") = res.v_thresh,
        Named("T_stop") = res.T_stop,
        Named("num_dummies") = res.num_dummies,
        Named("dummy_factor_L") = res.dummy_factor_L,
        Named("voting_grid") = res.voting_grid
    );
}

//' @title Get Selected Indices
//' @noRd
// [[Rcpp::export]]
IntegerVector trex_selector_get_selected_indices(XPtr<RTRexSelector> r_ptr) {
    auto indices = r_ptr->getSelectedIndices();
    IntegerVector res(indices.size());
    for(size_t i = 0; i < indices.size(); ++i) {
        res[i] = static_cast<int>(indices[i] + 1); // 1-based indexing for R
    }
    return res;
}


