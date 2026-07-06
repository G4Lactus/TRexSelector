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

        // The GVS control nests the base T-Rex control; merge the
        // separately-supplied trex_control into that nested field and derive
        // the solver_type GVS requires (EN -> TENET/TENET_AUG,
        // IEN -> TIENET_AUG), overriding whatever the caller left in
        // trex_control.solver_type.
        namespace gvs_sd = trex::trex_selector_methods::utils::solver_dispatch;
        TRexGVSControlParameter gvs_ctrl = gvs_control;
        gvs_ctrl.trex_ctrl = trex_control;
        if (gvs_ctrl.gvs_type == GVSType::IEN) {
            gvs_ctrl.trex_ctrl.solver_type = gvs_sd::SolverTypeForTRex::TIENET_AUG;
        } else { // EN
            gvs_ctrl.trex_ctrl.solver_type =
                (gvs_ctrl.en_solver == ENSolverType::TENET_AUG)
                    ? gvs_sd::SolverTypeForTRex::TENET_AUG
                    : gvs_sd::SolverTypeForTRex::TENET;
        }

        this->selector_ = std::make_unique<TRexGVSSelector>(
            *(this->X_map_), *(this->y_map_), tFDR, gvs_ctrl, seed, verbose
        );
    }

    TRexGVSSelector* get() const { return static_cast<TRexGVSSelector*>(this->selector_.get()); }
};

#endif // RCPP_TREX_GVS_H
