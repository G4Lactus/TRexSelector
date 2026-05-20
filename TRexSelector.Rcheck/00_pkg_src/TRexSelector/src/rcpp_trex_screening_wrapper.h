#ifndef RCPP_TREX_SCREENING_WRAPPER_H
#define RCPP_TREX_SCREENING_WRAPPER_H

#include <memory>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <trex_selector_methods/trex_screening/trex_screening.hpp>
#include "rcpp_trex_wrappers.h"

using namespace trex::trex_selector_methods::trex_screening;
using namespace trex::trex_selector_methods::trex_core;

/**
 * @brief Rcpp wrapper class for ScreenTRexSelector leveraging zero-copy references.
 * 
 * Exposes methods to parse R parameters and retrieve screening-centric results 
 * like bootstrap confidence levels and estimated FDR.
 */
class RTRexScreeningSelector : public RTRexSelector {
public:
    RTRexScreeningSelector(
        Eigen::Map<Eigen::MatrixXd> X,
        Eigen::Map<Eigen::VectorXd> y,
        const ScreenTRexControlParameter& screen_control,
        const tc::TRexControlParameter& trex_control,
        int seed,
        bool verbose
    ) {
        this->X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X);
        this->y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y);

        this->selector_ = std::make_unique<ScreenTRexSelector>(
            *(this->X_map_), *(this->y_map_), screen_control, trex_control, seed, verbose
        );
    }
    
    ScreenTRexSelector* get() const { return static_cast<ScreenTRexSelector*>(this->selector_.get()); }
};

#endif // RCPP_TREX_SCREENING_WRAPPER_H
