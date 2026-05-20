// ==============================================================================
// rcpp_trex_wrappers.h - Safe bindings wrappers for TRex objects
// ==============================================================================
#ifndef RCPP_TREX_WRAPPERS_H
#define RCPP_TREX_WRAPPERS_H
// ==============================================================================

// std includes
#include <memory>

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// TRex includes
#include <trex_selector_methods/trex_core/trex.hpp>

// ==============================================================================

using namespace trex::trex_selector_methods::trex_core;

// ==============================================================================

/**
 * @brief Rcpp wrapper class for TRexSelector leveraging zero-copy references.
 */
class RTRexSelector {
protected:
    RTRexSelector() = default;
public:
    virtual ~RTRexSelector() = default;
protected:
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
    virtual TRexSelector::SelectionResult select() { return selector_->select(); }

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

#endif // RCPP_TREX_WRAPPERS_H
