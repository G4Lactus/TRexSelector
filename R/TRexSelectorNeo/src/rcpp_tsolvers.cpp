// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <fstream>
#include <cereal/archives/portable_binary.hpp>
#include <tsolvers/tsolver_base.hpp>
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>
#include <tsolvers/linear_model/lars_based/tstepwise_solver.hpp>
#include <tsolvers/linear_model/lars_based/tenet_solver.hpp>
#include <tsolvers/linear_model/lars_based/tstagewise_solver.hpp>
#include <tsolvers/linear_model/omp_based/tomp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tgp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tacgp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tmp_solver.hpp>
#include <tsolvers/linear_model/omp_based/tools_solver.hpp>
#include <tsolvers/linear_model/omp_based/tncgmp_solver.hpp>
#include <tsolvers/linear_model/afs_based/tafs_solver.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>

// ==============================================================================

/*
 * Note on Roxygen Documentation:
 * All functions exported via Rcpp in this file are documented using standard Roxygen tags
 * but strictly include the `@noRd` tag. These are internal C++ bindings that are wrapped
 * by user-friendly R6 classes and R functions in the package namespace. Generating `.Rd`
 * manuals for these endpoints would clutter the public API.
 */

using namespace Rcpp;
using namespace trex::utils::memmap;
using namespace trex::tsolvers;
using namespace trex::tsolvers::linear_model::lars_based;
using namespace trex::tsolvers::linear_model::omp_based;
using namespace trex::tsolvers::linear_model::afs_based;

// ==============================================================================
// TSolverRcpp — stable data holder so Eigen::Map pointers never dangle
// ==============================================================================

/*
 * TSolver_Base stores raw Eigen::Map pointers (X_ and D_) that are set in the
 * constructor and must remain valid for the lifetime of the solver.  If we
 * passed Rcpp-converted Map objects directly as function-local variables those
 * pointers would become dangling the moment the Rcpp binding returned.
 *
 * TSolverRcpp owns heap-allocated Eigen::MatrixXd copies of X and D together
 * with stable Eigen::Map views over that data.  The solver is constructed with
 * references to those stable maps, so X_ and D_ always point to live memory.
 * This also fixes warm-start: after cereal deserialises the solver state
 * (which does not touch X_ / D_) the maps are still valid.
 */
struct TSolverRcpp {
    Eigen::MatrixXd              X_data_;   // owned copy of design matrix
    Eigen::MatrixXd              D_data_;   // owned copy of dummy matrix
    Eigen::Map<Eigen::MatrixXd>  X_map_;    // stable view over X_data_
    Eigen::Map<Eigen::MatrixXd>  D_map_;    // stable view over D_data_
    TSolver_Base*                solver_;   // the concrete solver (owned)

    // Member-initialisation order matches declaration order above, so
    // X_map_ / D_map_ are guaranteed to be initialised after the MatrixXd
    // members whose .data() they reference.
    TSolverRcpp(Eigen::MatrixXd X_in, Eigen::MatrixXd D_in)
        : X_data_(std::move(X_in)), D_data_(std::move(D_in)),
          X_map_(X_data_.data(), X_data_.rows(), X_data_.cols()),
          D_map_(D_data_.data(), D_data_.rows(), D_data_.cols()),
          solver_(nullptr) {}

    // Non-copyable and non-movable: moving would relocate X_data_ / D_data_
    // and invalidate the Map data pointers stored inside the solver.
    TSolverRcpp(const TSolverRcpp&)            = delete;
    TSolverRcpp& operator=(const TSolverRcpp&) = delete;
    TSolverRcpp(TSolverRcpp&&)                 = delete;
    TSolverRcpp& operator=(TSolverRcpp&&)      = delete;

    ~TSolverRcpp() { delete solver_; }
};

// ==============================================================================
// TSolver Instance Creators
// ==============================================================================

//' @title Create LARS Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_lars_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TLARS_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @title Create OMP Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_omp_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TOMP_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @title Create LARS Solver with Memory Mapped Matrix
//'
//' @param X_ptr Pointer to MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_lars_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TLARS_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @title Create OMP Solver with Memory Mapped Matrix
//'
//' @param X_ptr Pointer to MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//'
//' @return XPtr to TSolverRcpp
//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_omp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TOMP_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}

// ==============================================================================
// Additional TSolver Instance Creators (LARS family)
// ==============================================================================

//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tlasso_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TLASSO_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tstepwise_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TSTEPWISE_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tenet_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    double lambda2,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TENET_Solver(h->X_map_, h->D_map_, y, lambda2, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tstagewise_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TSTAGEWISE_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


// ==============================================================================
// Additional TSolver Instance Creators (OMP family)
// ==============================================================================

//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tgp_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TGP_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tacgp_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TACGP_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tmp_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TMP_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tools_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TOOLS_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tncgmp_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    int variant    = 0,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto v = static_cast<NCGMPVariant>(variant);
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TNCGMP_Solver(h->X_map_, h->D_map_, y, v, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


// ==============================================================================
// Additional TSolver Instance Creators (AFS family)
// ==============================================================================

//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tafs_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    double rho     = 1.0,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X), Eigen::MatrixXd(D));
    h->solver_ = new TAFS_Solver(h->X_map_, h->D_map_, y, rho, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


// ==============================================================================
// Additional TSolver Instance Creators (LARS family, mmap)
// ==============================================================================

//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tlasso_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TLASSO_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tstepwise_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TSTEPWISE_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tenet_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    double lambda2,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TENET_Solver(h->X_map_, h->D_map_, y, lambda2, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tstagewise_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TSTAGEWISE_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


// ==============================================================================
// Additional TSolver Instance Creators (OMP family, mmap)
// ==============================================================================

//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tgp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TGP_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tacgp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TACGP_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tmp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TMP_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tools_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TOOLS_Solver(h->X_map_, h->D_map_, y, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tncgmp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    int variant    = 0,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto v = static_cast<NCGMPVariant>(variant);
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TNCGMP_Solver(h->X_map_, h->D_map_, y, v, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


// ==============================================================================
// Additional TSolver Instance Creators (AFS family, mmap)
// ==============================================================================

//' @noRd
// [[Rcpp::export]]
XPtr<TSolverRcpp> tsolver_tafs_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y,
    double rho     = 1.0,
    bool normalize = true,
    bool intercept = true,
    bool verbose   = false
) {
    auto X_map = X_ptr->getMap();
    auto* h = new TSolverRcpp(Eigen::MatrixXd(X_map), Eigen::MatrixXd(D));
    h->solver_ = new TAFS_Solver(h->X_map_, h->D_map_, y, rho, normalize, intercept, verbose);
    return XPtr<TSolverRcpp>(h);
}


// ==============================================================================
// Common TSolver API endpoints (called via XPtr<TSolverRcpp>)
// ==============================================================================

//' @title Execute Step in TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @param T_stop Number of steps to compute
//' @param early_stop Whether to stop early based on constraints
//' @noRd
// [[Rcpp::export]]
void tsolver_execute_step(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    int T_stop,
    bool early_stop = true
) {
    // Treat -1 or any negative as full computation (max size_t)
    std::size_t c_T_stop = (T_stop < 0) ? -1 : static_cast<std::size_t>(T_stop);
    ptr->solver_->executeStep(c_T_stop, early_stop);
}


//' @title Get Beta from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @param step Step to retrieve beta for
//'
//' @return Beta vector
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd tsolver_get_beta(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    int step = -1
) {
    return ptr->solver_->getBeta(step);
}


//' @title Get Cp scores from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of Cp scores
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_cp(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    auto cp = ptr->solver_->getCp();
    return Rcpp::wrap(cp);
}


//' @title Get RSS from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of RSS scores
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_rss(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getRSS());
}


//' @title Get R2 from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of R2 scores
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_r2(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getR2());
}


//' @title Get Degrees of Freedom from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of DoF
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_dof(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getDoF());
}


//' @title Get Active Set from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of active variable indices
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_actives(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getActives());
}


//' @title Get Inactive Set from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @return Numeric vector of inactive variable indices
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_inactives(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->solver_->getInactives());
}


//' @title Get Intercept from TSolver
//'
//' @param ptr TSolverRcpp holder pointer
//' @param step Step to retrieve the intercept for
//' @return Numeric intercept value
//' @noRd
// [[Rcpp::export]]
double tsolver_get_intercept(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    int step = -1
) {
    return ptr->solver_->getIntercept(step);
}


//' @noRd
// [[Rcpp::export]]
int tsolver_get_num_steps(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return static_cast<int>(ptr->solver_->getNumSteps());
}


//' @noRd
// [[Rcpp::export]]
List tsolver_get_actions(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    const auto& actions = ptr->solver_->getActions();
    List result(actions.size());
    for (std::size_t i = 0; i < actions.size(); ++i)
        result[i] = Rcpp::wrap(actions[i]);
    return result;
}


// ==============================================================================
// Solver-Specific API Endpoints
// ==============================================================================

//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_lambda(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    if (auto* p = dynamic_cast<TLARS_Solver*>(ptr->solver_)) {
        return Rcpp::wrap(p->getLambda());
    }
    stop("tsolver_get_lambda: Solver type does not have a lambda path.");
}


//' @noRd
// [[Rcpp::export]]
int tsolver_get_num_removals(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    if (auto* p = dynamic_cast<TSTAGEWISE_Solver*>(ptr->solver_)) {
        return static_cast<int>(p->getNumRemovals());
    }
    if (auto* p = dynamic_cast<TENET_Solver*>(ptr->solver_)) {
        return static_cast<int>(p->getNumRemovals());
    }
    if (auto* p = dynamic_cast<TLASSO_Solver*>(ptr->solver_)) {
        return static_cast<int>(p->getNumRemovals());
    }
    stop("tsolver_get_num_removals: Solver type does not track removals.");
}


//' @noRd
// [[Rcpp::export]]
double tsolver_get_cycling_ratio(
    XPtr<TSolverRcpp> ptr // NOLINT(performance-unnecessary-value-param)
) {
    if (auto* p = dynamic_cast<TSTAGEWISE_Solver*>(ptr->solver_)) {
        return p->getCyclingRatio();
    }
    if (auto* p = dynamic_cast<TENET_Solver*>(ptr->solver_)) {
        return p->getCyclingRatio();
    }
    stop("tsolver_get_cycling_ratio: Solver type does not support getCyclingRatio().");
}


//' @title Save TSolver to File
//'
//' @param ptr TSolverRcpp holder pointer
//' @param filename Output filename
//' @noRd
// [[Rcpp::export]]
void tsolver_save(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        stop("tsolver_save: Cannot open file for saving.");
    }
    cereal::PortableBinaryOutputArchive archive(os);

    TSolver_Base* s = ptr->solver_;
    // Derived classes must be checked before their base classes.
    // LARS family
    if (auto* p = dynamic_cast<TSTAGEWISE_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TLASSO_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TSTEPWISE_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TLARS_Solver*>(s)) {
        archive(*p);
    // OMP family
    } else if (auto* p = dynamic_cast<TACGP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TGP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TMP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TOOLS_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TOMP_Solver*>(s)) {
        archive(*p);
    // Independent families
    } else if (auto* p = dynamic_cast<TNCGMP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TAFS_Solver*>(s)) {
        archive(*p);
    } else {
        stop("tsolver_save: Unsupported TSolver type.");
    }
}


//' @title Load TSolver from File
//'
//' @param ptr TSolverRcpp holder pointer
//' @param filename Input filename
//' @noRd
// [[Rcpp::export]]
void tsolver_load(
    XPtr<TSolverRcpp> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        stop("tsolver_load: Cannot open file for loading.");
    }
    cereal::PortableBinaryInputArchive archive(is);

    TSolver_Base* s = ptr->solver_;
    // Derived classes must be checked before their base classes.
    // LARS family
    if (auto* p = dynamic_cast<TSTAGEWISE_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TENET_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TLASSO_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TSTEPWISE_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TLARS_Solver*>(s)) {
        archive(*p);
    // OMP family
    } else if (auto* p = dynamic_cast<TACGP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TGP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TMP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TOOLS_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TOMP_Solver*>(s)) {
        archive(*p);
    // Independent families
    } else if (auto* p = dynamic_cast<TNCGMP_Solver*>(s)) {
        archive(*p);
    } else if (auto* p = dynamic_cast<TAFS_Solver*>(s)) {
        archive(*p);
    } else {
        stop("tsolver_load: Unsupported TSolver type.");
    }
    // X_ and D_ are not serialised; reconnect to the holder's stable maps.
    ptr->solver_->reconnect(ptr->X_map_, ptr->D_map_);
}
