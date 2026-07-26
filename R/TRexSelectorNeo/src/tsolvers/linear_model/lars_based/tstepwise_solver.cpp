// =============================================================================
// tstepwise_solver.cpp
// =============================================================================
/**
 * @file tstepwise_solver.cpp
 * @brief Implementation of TSTEPWISE_Solver, a T-LARS-based stepwise
 * regression solver. It is mathematically equivalent to OMP.
 * The method performs no variable removals.
 */
// =============================================================================

// std includes
#include <fstream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tstepwise_solver.hpp>


// ============================================================================

// Embedded into namespace trex::tsolvers::linear_model::lars_based
namespace trex::tsolvers::linear_model::lars_based {

// ============================================================================
// Core Algorithm Implementation: executeStep()
// ============================================================================

void TSTEPWISE_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    // Set openMP dominance for the entire algorithm
    EigenSingleThreadGuard eigen_single_thread_guard;

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < effective_n_ &&
        (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)
    ) {

        // ========================================================
        // STEP 1: Use pre-computed correlations
        // ========================================================
        auto [Cmax, new_vars] = findTiedMaxCorrelations();

        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0 or no variables to add; exiting.");
            break;
        }

        pruneTiedDummies(new_vars, T_stop, early_stop);

        // ========================================================
        // STEP 2: Increment step counter and record lambda
        // ========================================================
        currentStep_++;
        lambda_.push_back(Cmax);

        // ========================================================
        // STEP 3: Active set update
        // ========================================================
        actions_.emplace_back(updateActiveSet(new_vars));

        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; aborting.");
            // Roll back the aborted step so actions_/lambda_ stay aligned
            // with the diagnostics arrays (rejections remain recorded in
            // dropped_indices_) and rebuild inactives_ so dropped columns
            // cannot be re-selected.
            actions_.pop_back();
            lambda_.pop_back();
            currentStep_--;
            updateInactiveSet();
            break;
        }

        // ========================================================
        // STEP 4: Full step to axis-aligned solution
        // ========================================================
        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);
        Eigen::VectorXd u = equiangularVector(w_A);

        // T-Stepwise takes full orthogonal step (reduces correlations to zero)
        double gamma = Cmax / A_A_;

        // ========================================================
        // STEP 5: Update beta coefficients and residuals
        // ========================================================
        updateBetaPath(w_A, gamma);
        recordBetaStep();
        r_ -= gamma * u;

        // ========================================================
        // STEP 6: Update diagnostics
        // ========================================================
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        // Track dummy count and compute DoF (includes dummies + intercept)
        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 7: Update inactive set and recompute correlations
        // ========================================================
        // Update inactives first because we completely recompute correlations
        // from the residual (unlike TLARS/TLASSO which do incremental updates)
        updateInactiveSet();
        zeroAllSigns();
        recomputeCorrelations();
    }
}


// ============================================================================
// Internal Helper
// ============================================================================

void TSTEPWISE_Solver::zeroAllSigns() {
    // Reset signs to zero for orthogonal (axis-aligned) steps.
    // This ensures each new variable enters with fresh correlation sign.
    Sign_.assign(Sign_.size(), 0);
}

void TSTEPWISE_Solver::recomputeCorrelations() {
    // Full recomputation X^T r over inactives (shared base helper)
    refreshInactiveCorrelations();
}


// ============================================================================
// Serialization Implementation
// ============================================================================

void TSTEPWISE_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TSTEPWISE_Solver TSTEPWISE_Solver::load(const std::string& filename,
                                        Eigen::Map<Eigen::MatrixXd>& X,
                                        Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TSTEPWISE_Solver>(filename, X, D);
}

// ============================================================================
} /* End of namespace trex::tsolvers::linear_model::lars_based */
