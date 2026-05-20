// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <fstream>
#include <cereal/archives/portable_binary.hpp>
#include <tsolvers/tsolver_base.hpp>
#include <tsolvers/linear_model/lars_based/tlars_solver.hpp>
#include <tsolvers/linear_model/omp_based/tomp_solver.hpp>
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

// ==============================================================================
// TSolver Instance Creators
// ==============================================================================

//' @title Create LARS Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//'
//' @return XPtr to TSolver_Base
//' @noRd
// [[Rcpp::export]]
XPtr<TSolver_Base> tsolver_lars_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y
) {
    return XPtr<TSolver_Base>(new TLARS_Solver(X, D, y));
}


//' @title Create OMP Solver
//'
//' @param X Design matrix
//' @param D Dummy matrix
//' @param y Response vector
//'
//' @return XPtr to TSolver_Base
//' @noRd
// [[Rcpp::export]]
XPtr<TSolver_Base> tsolver_omp_create(
    Eigen::Map<Eigen::MatrixXd> X,
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y
) {
    return XPtr<TSolver_Base>(new TOMP_Solver(X, D, y));
}


//' @title Create LARS Solver with Memory Mapped Matrix
//'
//' @param X_ptr Pointer to MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//'
//' @return XPtr to TSolver_Base
//' @noRd
// [[Rcpp::export]]
XPtr<TSolver_Base> tsolver_lars_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y
) {
    auto X_map = X_ptr->getMap();
    return XPtr<TSolver_Base>(new TLARS_Solver(X_map, D, y));
}


//' @title Create OMP Solver with Memory Mapped Matrix
//'
//' @param X_ptr Pointer to MemoryMappedMatrix
//' @param D Dummy matrix
//' @param y Response vector
//'
//' @return XPtr to TSolver_Base
//' @noRd
// [[Rcpp::export]]
XPtr<TSolver_Base> tsolver_omp_mmap_create(
    XPtr<MemoryMappedMatrix<double>> X_ptr, // NOLINT(performance-unnecessary-value-param)
    Eigen::Map<Eigen::MatrixXd> D,
    Eigen::Map<Eigen::VectorXd> y
) {
    auto X_map = X_ptr->getMap();
    return XPtr<TSolver_Base>(new TOMP_Solver(X_map, D, y));
}

// ==============================================================================
// Common TSolver API endpoints (called via XPtr<TSolver_Base>)
// ==============================================================================

//' @title Execute Step in TSolver
//'
//' @param ptr TSolver_Base instance pointer
//' @param T_stop Number of steps to compute
//' @param early_stop Whether to stop early based on constraints
//' @noRd
// [[Rcpp::export]]
void tsolver_execute_step(
    XPtr<TSolver_Base> ptr, // NOLINT(performance-unnecessary-value-param)
    int T_stop,
    bool early_stop = true
) {
    // Treat -1 or any negative as full computation (max size_t)
    std::size_t c_T_stop = (T_stop < 0) ? -1 : static_cast<std::size_t>(T_stop);
    ptr->executeStep(c_T_stop, early_stop);
}


//' @title Get Beta from TSolver
//'
//' @param ptr TSolver_Base instance pointer
//' @param step Step to retrieve beta for
//'
//' @return Beta vector
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd tsolver_get_beta(
    XPtr<TSolver_Base> ptr, // NOLINT(performance-unnecessary-value-param)
    int step = -1
) {
    return ptr->getBeta(step);
}


//' @title Get Cp scores from TSolver
//'
//' @param ptr TSolver_Base instance pointer
//' @return Numeric vector of Cp scores
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_cp(
    XPtr<TSolver_Base> ptr // NOLINT(performance-unnecessary-value-param)
) {
    auto cp = ptr->getCp();
    return Rcpp::wrap(cp);
}


//' @title Get RSS from TSolver
//'
//' @param ptr TSolver_Base instance pointer
//' @return Numeric vector of RSS scores
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_rss(
    XPtr<TSolver_Base> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->getRSS());
}


//' @title Get R2 from TSolver
//'
//' @param ptr TSolver_Base instance pointer
//' @return Numeric vector of R2 scores
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_r2(
    XPtr<TSolver_Base> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->getR2());
}


//' @title Get Degrees of Freedom from TSolver
//'
//' @param ptr TSolver_Base instance pointer
//' @return Numeric vector of DoF
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_dof(
    XPtr<TSolver_Base> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->getDoF());
}


//' @title Get Active Set from TSolver
//'
//' @param ptr TSolver_Base instance pointer
//' @return Numeric vector of active variable indices
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_actives(
    XPtr<TSolver_Base> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->getActives());
}


//' @title Get Inactive Set from TSolver
//'
//' @param ptr TSolver_Base instance pointer
//' @return Numeric vector of inactive variable indices
//' @noRd
// [[Rcpp::export]]
NumericVector tsolver_get_inactives(
    XPtr<TSolver_Base> ptr // NOLINT(performance-unnecessary-value-param)
) {
    return Rcpp::wrap(ptr->getInactives());
}


//' @title Get Intercept from TSolver
//'
//' @param ptr TSolver_Base instance pointer
//' @param step Step to retrieve the intercept for
//' @return Numeric intercept value
//' @noRd
// [[Rcpp::export]]
double tsolver_get_intercept(
    XPtr<TSolver_Base> ptr, // NOLINT(performance-unnecessary-value-param)
    int step = -1
) {
    return ptr->getIntercept(step);
}


//' @title Save TSolver to File
//'
//' @param ptr TSolver_Base instance pointer
//' @param filename Output filename
//' @noRd
// [[Rcpp::export]]
void tsolver_save(
    XPtr<TSolver_Base> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        stop("tsolver_save: Cannot open file for saving.");
    }
    cereal::PortableBinaryOutputArchive archive(os);

    if (auto lars = dynamic_cast<TLARS_Solver*>(ptr.get())) {
        archive(*lars);
    } else if (auto omp = dynamic_cast<TOMP_Solver*>(ptr.get())) {
        archive(*omp);
    } else {
        stop("tsolver_save: Unsupported TSolver type.");
    }
}


//' @title Load TSolver from File
//'
//' @param ptr TSolver_Base instance pointer
//' @param filename Input filename
//' @noRd
// [[Rcpp::export]]
void tsolver_load(
    XPtr<TSolver_Base> ptr, // NOLINT(performance-unnecessary-value-param)
    const std::string& filename
) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        stop("tsolver_load: Cannot open file for loading.");
    }
    cereal::PortableBinaryInputArchive archive(is);

    if (auto lars = dynamic_cast<TLARS_Solver*>(ptr.get())) {
        archive(*lars);
    } else if (auto omp = dynamic_cast<TOMP_Solver*>(ptr.get())) {
        archive(*omp);
    } else {
        stop("tsolver_load: Unsupported TSolver type.");
    }
}
