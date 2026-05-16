// ==============================================================================
// rcpp_trex_wrappers.h - Safe bindings wrappers for TRex objects
// ==============================================================================
#ifndef TREX_RCPP_TREX_WRAPPERS_H
#define TREX_RCPP_TREX_WRAPPERS_H

#include <memory>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <trex_selector_methods/trex_core/trex.hpp>

using namespace trex::trex_selector_methods::trex_core;

/**
 * @brief Rcpp wrapper class for TRexSelector leveraging zero-copy references.
 */
class RTRexSelector {
private:
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_map_;
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_map_;
    std::unique_ptr<TRexSelector> selector_;

public:
    RTRexSelector(
        Eigen::Map<Eigen::MatrixXd> X,
        Eigen::Map<Eigen::VectorXd> y,
        double tFDR,
        const TRexControlParameter& trex_control,
        int seed,
        bool verbose
    ) {
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X);
        y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y);

        selector_ = std::make_unique<TRexSelector>(
            *X_map_, *y_map_, tFDR, trex_control, seed, verbose
        );
    }
    
    TRexSelector* get() const { return selector_.get(); }
    TRexSelector::SelectionResult select() { return selector_->select(); }

    // ============================================================
    // Public Getters
    // ============================================================
    double getTFDR() const noexcept { return selector_->getTFDR(); }
    std::size_t getK() const noexcept { return selector_->getK(); }
    std::size_t getMaxNumDummies() const noexcept { return selector_->getMaxNumDummies(); }
    bool getMaxTStop() const noexcept { return selector_->getMaxTStop(); }
    bool getStagnationCheck() const noexcept { return selector_->getStagnationCheck(); }

    const Eigen::VectorXi& getSelectedVar() const noexcept { return selector_->getSelectedVar(); }
    const std::vector<std::size_t>& getSelectedIndices() const noexcept { return selector_->getSelectedIndices(); }
    std::size_t getNumSelected() const noexcept { return selector_->getNumSelected(); }
    std::size_t getTStop() const noexcept { return selector_->getTStop(); }
    const Eigen::VectorXd& getVotingGrid() const noexcept { return selector_->getVotingGrid(); }
    double getVotingThreshold() const noexcept { return selector_->getVotingThreshold(); }
};


#include <trex_selector_methods/trex_da/trex_da.hpp>
using namespace trex::trex_selector_methods::trex_da;

/**
 * @brief Rcpp wrapper class for TRexDASelector leveraging zero-copy references.
 */
class RTRexDASelector {
private:
    std::unique_ptr<Eigen::Map<Eigen::MatrixXd>> X_map_;
    std::unique_ptr<Eigen::Map<Eigen::VectorXd>> y_map_;
    std::unique_ptr<TRexDASelector> selector_;

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
        X_map_ = std::make_unique<Eigen::Map<Eigen::MatrixXd>>(X);
        y_map_ = std::make_unique<Eigen::Map<Eigen::VectorXd>>(y);

        selector_ = std::make_unique<TRexDASelector>(
            *X_map_, *y_map_, tFDR, da_control, trex_control, seed, verbose
        );
    }
    
    TRexDASelector* get() const { return selector_.get(); }
    const TRexDASelector::DASelectionResult& select() { 
        selector_->select(); 
        return selector_->getDAResult();
    }
};

#endif // TREX_RCPP_TREX_WRAPPERS_H

