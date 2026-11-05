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
#include "rcpp_trex_tikhonov.h"
#include "rcpp_trex_screening.h"
#include "rcpp_trex_biobank_screening.h"

// ===============================================================================

// Forward declare or include all classes used in XPtr by RcppExports
#include <ml_methods/scaler_methods/z_score_scaler.hpp>
#include <ml_methods/scaler_methods/lp_norm_scaler.hpp>
#include <ml_methods/model_selection/ridge_cv_svd.hpp>
#include <ml_methods/model_selection/enet_cv_ccd.hpp>
#include <ml_methods/svd/svd.hpp>
#include <ml_methods/pca/pca.hpp>
#include <ml_methods/ridge_regression/ridge.hpp>
#include <tsolvers/ad/tsolver_base.hpp>
#include <utils/memmap/memory_mapped_matrix.hpp>

// ===============================================================================
// TSolverRcpp — zero-copy holder of the T-solver and its views
// ===============================================================================
//
// The single definition, shared by rcpp_tsolvers.cpp and RcppExports.cpp (which
// instantiates the XPtr delete-finalizer from it).
//
// The maps view the caller's memory directly — the R matrix (REALSXP, no copy)
// or the memory-mapped file — and the XPtr's protection slot pins the R objects
// (X, D, y, or the mmap holder) until the finalizer has run. The solver keeps
// its own copies of the maps; the holder keeps them for $restore() and $load().
//
// Nothing is copied: a solver constructed with normalize = TRUE or intercept =
// TRUE preprocesses the R matrix (or the mmap file) IN PLACE and holds a claim
// on it until $restore() or garbage collection; with both FALSE it is a reader
// and never writes (TSolver_Base::restore() documents the contract). y is never
// written: the solver centers an internal copy.

struct TSolverRcpp {
    Eigen::Map<Eigen::MatrixXd>  X_map_;    // view over the R matrix or the mmap file
    Eigen::Map<Eigen::MatrixXd>  D_map_;    // view over the R matrix
    Eigen::Map<Eigen::VectorXd>  y_map_;    // view over the R vector (never written)
    trex::tsolvers::ad::TSolver_Base* solver_;   // the concrete solver (owned)

    TSolverRcpp(Eigen::Map<Eigen::MatrixXd> X,
                Eigen::Map<Eigen::MatrixXd> D,
                Eigen::Map<Eigen::VectorXd> y)
        : X_map_(X), D_map_(D), y_map_(y), solver_(nullptr) {}

    TSolverRcpp(const TSolverRcpp&)            = delete;
    TSolverRcpp& operator=(const TSolverRcpp&) = delete;
    TSolverRcpp(TSolverRcpp&&)                 = delete;
    TSolverRcpp& operator=(TSolverRcpp&&)      = delete;

    // A writer that is garbage-collected without $restore() writes the
    // caller's X and D back from ~TSolver_Base (the finalizer safety net),
    // so the R matrix or the mmap file never stays preprocessed without an
    // owner. The R objects are still alive then (protection slot).
    ~TSolverRcpp() { delete solver_; }
};

#endif

using namespace trex::ml_methods::scaler_methods;
using namespace trex::ml_methods::model_selection;

// The ridge_cv_* endpoints are backed by the SVD-path implementation
// (the former ridge_cv and the GCV variant were removed from the core).
using ridge_cv = trex::ml_methods::model_selection::ridge_cv_svd;

// Elastic-net (coordinate-descent) endpoints: enet_path is the glmnet-style
// path fit, enet_cv its K-fold CV lambda selector.
using enet_path = trex::ml_methods::model_selection::enet_gaussian;
using enet_cv   = trex::ml_methods::model_selection::enet_cv_ccd;

// Memory-mapped matrix handle shared by the mmap_matrix() utilities and the
// zero-copy *_mmap ml_methods endpoints.
using MmapMatrixD = trex::utils::memmap::MemoryMappedMatrix<double>;
using namespace trex::ml_methods::svd;
using namespace trex::ml_methods::pca;
using namespace trex::ml_methods::ridge;
using namespace trex::tsolvers::ad;
using namespace trex::utils::memmap;

#ifdef FALSE
#undef FALSE
#endif
#define FALSE ((Rboolean)0)
