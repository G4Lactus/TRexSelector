// ==============================================================================
// rcpp_trex_biobank_screening_wrapper.h
// ==============================================================================
#ifndef RCPP_TREX_BIOBANK_SCREENING_WRAPPER_H
#define RCPP_TREX_BIOBANK_SCREENING_WRAPPER_H
// ==============================================================================

#include <memory>

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#include <trex_selector_methods/trex_screening/trex_biobank_screening.hpp>

// ==============================================================================

using namespace trex::trex_selector_methods::trex_core;
using namespace trex::trex_selector_methods::trex_screening;
using namespace trex::trex_selector_methods::trex_biobank_screening;

// ==============================================================================

/**
 * @brief Rcpp wrapper class for BiobankScreenTRex leveraging zero-copy references.
 */
class RTRexBiobankScreeningSelector {
private:
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_map_;
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_map_1d_;
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> Y_map_2d_;
    std::unique_ptr<BiobankScreenTRex> selector_;

public:
    // 1D Constructor (single phenotype)
    RTRexBiobankScreeningSelector(
        Eigen::Map<Eigen::MatrixXd> X,
        Eigen::Map<Eigen::VectorXd> y,
        const BiobankScreenTRexControl& biosctrex_ctrl,
        int seed,
        bool verbose
    ) {
        this->X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X);
        this->y_map_1d_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y);

        this->selector_ = std::make_unique<BiobankScreenTRex>(
            *(this->X_map_), *(this->y_map_1d_), biosctrex_ctrl, seed, verbose
        );
    }

    // 2D Constructor (multiple phenotypes)
    RTRexBiobankScreeningSelector(
        Eigen::Map<Eigen::MatrixXd> X,
        Eigen::Map<Eigen::MatrixXd> Y,
        const BiobankScreenTRexControl& biosctrex_ctrl,
        int seed,
        bool verbose
    ) {
        this->X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X);
        this->Y_map_2d_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(Y);

        this->selector_ = std::make_unique<BiobankScreenTRex>(
            *(this->X_map_), *(this->Y_map_2d_), biosctrex_ctrl, seed, verbose
        );
    }

    BiobankScreenTRex* get() const { return this->selector_.get(); }
};

// ==============================================================================
#endif /* End of RCPP_TREX_BIOBANK_SCREENING_WRAPPER_H */
