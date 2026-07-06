// ===============================================================================
// TRexSlector_types.h - Type definitions for Rcpp bindings of TRexSelector
// ===============================================================================
#ifndef RCPP_TREX_SELECTOR_TYPES_H
#define RCPP_TREX_SELECTOR_TYPES_H
// ===============================================================================
/**
 * @file TRexSelector_types.h
 *
 * @brief Special file for rcpp to look up all types used in XPtr across multiple
 *        source files. The file serves as a master aggregator that explicitly
 *        includes and binds namespaces.
 */
// ===============================================================================

// Rcpp includes
#include <RcppEigen.h>
#include <Rcpp.h>

// ===============================================================================

// Full definitions are required (not forward declarations): RcppExports.cpp
// instantiates the XPtr delete-finalizer for these types, and deleting through
// an incomplete type would skip the destructors of the held resources.
#include "rcpp_trex_selector.h"
#include "rcpp_trex_da.h"
#include "rcpp_trex_gvs.h"
#include "rcpp_trex_screening.h"
#include "rcpp_trex_biobank_screening.h"

// ===============================================================================

// Forward declare or include all classes used in XPtr by RcppExports
#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>
#include <ml_methods/model_selection/ridge_cv_svd.hpp>
#include <ml_methods/svd/svd.hpp>
#include <ml_methods/pca/pca.hpp>
#include <ml_methods/ridge_regression/ridge.hpp>
#include <tsolvers/tsolver_base.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>

// ===============================================================================
// TSolverRcpp — stable data holder so Eigen::Map pointers never dangle
// ===============================================================================

struct TSolverRcpp {
    Eigen::MatrixXd              X_data_;
    Eigen::MatrixXd              D_data_;
    Eigen::Map<Eigen::MatrixXd>  X_map_;
    Eigen::Map<Eigen::MatrixXd>  D_map_;
    trex::tsolvers::TSolver_Base* solver_;

    TSolverRcpp(Eigen::MatrixXd X_in, Eigen::MatrixXd D_in)
        : X_data_(std::move(X_in)), D_data_(std::move(D_in)),
          X_map_(X_data_.data(), X_data_.rows(), X_data_.cols()),
          D_map_(D_data_.data(), D_data_.rows(), D_data_.cols()),
          solver_(nullptr) {}

    TSolverRcpp(const TSolverRcpp&)            = delete;
    TSolverRcpp& operator=(const TSolverRcpp&) = delete;
    TSolverRcpp(TSolverRcpp&&)                 = delete;
    TSolverRcpp& operator=(TSolverRcpp&&)      = delete;

    ~TSolverRcpp() { delete solver_; }
};

#endif

using namespace trex::ml_methods::scaler_methods;
using namespace trex::ml_methods::model_selection;

// The ridge_cv_* endpoints are backed by the SVD-path implementation
// (the former ridge_cv and the GCV variant were removed from the core).
using ridge_cv = trex::ml_methods::model_selection::ridge_cv_svd;
using namespace trex::ml_methods::svd;
using namespace trex::ml_methods::pca;
using namespace trex::ml_methods::ridge;
using namespace trex::tsolvers;
using namespace trex::utils::memmap;

#ifdef FALSE
#undef FALSE
#endif
#define FALSE ((Rboolean)0)
