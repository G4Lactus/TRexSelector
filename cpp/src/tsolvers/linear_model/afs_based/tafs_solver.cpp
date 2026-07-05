// ===================================================================================
// tafs_solver.cpp
// ===================================================================================

// std includes
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

// tsolvers includes
#include "tafs_solver.hpp"

// utils includes
#include <utils/openmp/utils_openmp.hpp>

// ==================================================================================

namespace trex::tsolvers::linear_model::afs_based {

// ==================================================================================
// Constructor Implementation
// ==================================================================================

TAFS_Solver::TAFS_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    double rho,
    bool normalize,
    bool intercept,
    bool verbose,
    ScalingMode scaling_mode)
    : TSolver_Base(X, D, y, normalize, intercept, verbose, scaling_mode), rho_(rho) {

    if (rho_ <= 0.0 || rho_ > 1.0) {
        throw std::invalid_argument("TAFS_Solver: rho must be in the range (0, 1].");
    }

    dropped_indices_.clear();
    if (normalize_ || intercept_) { preprocess(); }
    initializeInactives();

    std::size_t p_tot = p_original_ + num_dummies_;
    max_actives_ = std::min(p_tot - dropped_indices_.size(), effective_n_);

    // With shrinkage each step moves only rho of the way to the OLS fit, so
    // convergence needs on the order of 1/rho more steps than pure stepwise.
    maxSteps_ = static_cast<std::size_t>(std::ceil(
        8.0 * static_cast<double>(std::min(max_actives_, effective_n_)) / rho_));

    logInfo(concatMsg(
            solverTypeToString(), " sequence\n",
            "  Samples (n): ", X.rows(), "\n",
            "  Features (p): ", p_tot, "\n",
            "  Shrinkage (rho): ", rho_, "\n",
            "  Max steps: ", maxSteps_, "\n"
    ));

    r_ = y_;
    betaPath_.resize(static_cast<Eigen::Index>(p_tot), 1);
    betaPath_.col(0).setZero();

    double rss = y_.dot(y_);
    RSS_.emplace_back(rss);
    R2_.emplace_back(0.0);
    DoF_.emplace_back(intercept_ ? 1 : 0);
    dummies_at_step_.reserve(maxSteps_ + 1);
    dummies_at_step_.push_back(0);

    initializeCorrelations();
}

TAFS_Solver::TAFS_Solver() : TSolver_Base() {}

// ============================================================================
// Core Algorithm Implementation
// ============================================================================

void TAFS_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();
    EigenSingleThreadGuard eigen_single_thread_guard;

    while (currentStep_ < maxSteps_ && !inactives_.empty() &&
           actives_.size() < effective_n_ &&
           (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)) {

        // 1. Max Correlation Screening
        // -----------------------------------
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0 or no variables to add; exiting.");
            break;
        }
        pruneTiedDummies(new_vars, T_stop, early_stop);

        currentStep_++;
        std::vector<int> actions_this_step;
        any_dropped_ = false;


        // 2. Active Set & Cholesky Update
        // -----------------------------------
        for (auto j_new : new_vars) {
            // updateR never mutates R_, so only the rank needs restoring on rejection
            int rankR_backup = last_updateR_rank_;

            Eigen::MatrixXd newR = updateR(getColumn(j_new), eps_);

            if (last_updateR_rank_ == static_cast<int>(actives_.size())) {
                last_updateR_rank_ = rankR_backup;
                dropped_indices_.push_back(j_new);
                actions_this_step.push_back(-static_cast<int>(j_new));
                any_dropped_ = true;
                logWarning(concatMsg("Variable ", j_new, " collinear; dropped."));
            } else {
                R_ = newR;
                actives_.push_back(j_new);
                actions_this_step.push_back(static_cast<int>(j_new));
                num_additions_++;
                std::erase(inactives_, j_new);
                if (j_new >= dummy_start_idx_) count_active_dummies_++;
            }
        }
        actions_.push_back(actions_this_step);

        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; exiting.");
            // Roll back the aborted step so actions_ stays aligned with the
            // diagnostics arrays (rejections remain in dropped_indices_) and
            // rebuild inactives_ so dropped columns cannot be re-selected.
            actions_.pop_back();
            currentStep_--;
            updateInactiveSet();
            break;
        }


        // 3. OLS Estimation on the Active Set
        // -----------------------------------
        std::size_t k = actives_.size();
        Eigen::VectorXd b(k);
        for (std::size_t i = 0; i < k; ++i) {
            b(static_cast<Eigen::Index>(i)) = getColumn(actives_[i]).dot(y_);
        }

        Eigen::VectorXd u = backsolveT(R_, b);
        Eigen::VectorXd nu = backsolve(R_, u); // OLS coefficients

        Eigen::Index p_tot = static_cast<Eigen::Index>(p_original_ + num_dummies_);
        Eigen::VectorXd beta_hat = Eigen::VectorXd::Zero(p_tot);
        for (std::size_t i = 0; i < k; ++i) {
            beta_hat(static_cast<Eigen::Index>(actives_[i])) =
                nu(static_cast<Eigen::Index>(i));
        }


        // 4. Blending / Shrinkage
        // -----------------------------------
        ensureBetaPathCapacity(currentStep_ + 1);

        Eigen::VectorXd beta_prev = betaPath_.col(static_cast<Eigen::Index>(currentStep_ - 1));
        Eigen::VectorXd beta_blended = (1.0 - rho_) * beta_prev + rho_ * beta_hat;
        betaPath_.col(static_cast<Eigen::Index>(currentStep_)) = beta_blended;


        // 5. Update Residuals
        // -----------------------------------
        r_ = y_;
        for (std::size_t j : actives_) {
            r_ -= getColumn(j) * beta_blended(static_cast<Eigen::Index>(j));
        }


        // 6. Diagnostics
        // -----------------------------------
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));


        // 7. Update Inactives and Correlations
        // -----------------------------------
        updateInactiveSet();
        updateCorrelations();
    }

    // Drop any spare beta-path capacity now that execution stopped
    trimBetaPathToRecordedSteps();
}

// ==================================================================================
// Math Helpers
// ==================================================================================

void TAFS_Solver::updateCorrelations() {
    refreshInactiveCorrelations();
}

// ==================================================================================
// Serialization
// ==================================================================================

void TAFS_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }

TAFS_Solver TAFS_Solver::load(const std::string& filename,
                              Eigen::Map<Eigen::MatrixXd>& X,
                              Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TAFS_Solver>(filename, X, D);
}

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::afs_based */
// ==================================================================================
