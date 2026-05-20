// ==============================================================================
#ifndef RCPP_TREX_GVS_H
#define RCPP_TREX_GVS_H
// ==============================================================================

// std includes
#include <memory>

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// TRex includes
#include <trex_selector_methods/trex_gvs/trex_gvs.hpp>
#include "rcpp_trex_selector.h"

// ==============================================================================

using namespace trex::trex_selector_methods::trex_gvs;
using namespace trex::trex_selector_methods::trex_core;

// ==============================================================================

/**
 * @brief Rcpp wrapper class for TRexGVSSelector leveraging zero-copy references.
 * 
 * Maps R objects (vectors, matrices, lists) to the corresponding C++ structures
 * for the Dependency-Aware Group Variable Selection T-Rex variant.
 */
class RTRexGVSSelector : public RTRexSelector {
public:
    RTRexGVSSelector(
        Eigen::Map<Eigen::MatrixXd> X,
        Eigen::Map<Eigen::VectorXd> y,
        double tFDR,
        const TRexGVSControlParameter& gvs_control,
        const tc::TRexControlParameter& trex_control,
        int seed,
        bool verbose
    ) {
        this->X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X);
        this->y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y);

        this->selector_ = std::make_unique<TRexGVSSelector>(
            *(this->X_map_), *(this->y_map_), tFDR, gvs_control, trex_control, seed, verbose
        );
    }

    TRexGVSSelector* get() const { return static_cast<TRexGVSSelector*>(this->selector_.get()); }
};

#endif // RCPP_TREX_GVS_H
