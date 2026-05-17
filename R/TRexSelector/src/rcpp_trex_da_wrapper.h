// ==============================================================================
#ifndef RCPP_TREX_DA_WRAPPER_H
#define RCPP_TREX_DA_WRAPPER_H
// ==============================================================================

// std includes
#include <memory>

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// TRex includes
#include <trex_selector_methods/trex_da/trex_da.hpp>
#include "rcpp_trex_wrappers.h"

// ==============================================================================

using namespace trex::trex_selector_methods::trex_da;
using namespace trex::trex_selector_methods::trex_core;

// ==============================================================================
/**
 * @brief Rcpp wrapper class for TRexDASelector leveraging zero-copy references.
 */
class RTRexDASelector : public RTRexSelector {
public:
    RTRexDASelector(
        Eigen::Map<Eigen::MatrixXd> X,
        Eigen::Map<Eigen::VectorXd> y,
        double tFDR,
        const TRexDAControlParameter& da_control,
        const TRexControlParameter& trex_control,
        int seed,
        bool verbose
    ) {
        this->X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X);
        this->y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y);

        this->selector_ = std::make_unique<TRexDASelector>(
            *(this->X_map_), *(this->y_map_), tFDR, da_control, trex_control, seed, verbose
        );
    }

    TRexDASelector* get() const { return static_cast<TRexDASelector*>(this->selector_.get()); }
};

#endif // RCPP_TREX_DA_WRAPPER_H
