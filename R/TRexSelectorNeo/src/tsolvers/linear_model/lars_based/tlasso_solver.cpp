// ============================================================================
// tlasso_solver.cpp
// ============================================================================
/**
 * @file tlasso_solver.cpp
 *
 * @brief Implementation of the Terminating LASSO (T-LASSO) solver class,
 * extending T-LARS solver.
 *
 */
// ============================================================================

// std includes
#include <fstream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/lars_based/tlasso_solver.hpp>


// ============================================================================

// Embedded into namespace trex::tsolvers::linear_model::lars_based
namespace trex::tsolvers::linear_model::lars_based {

// ============================================================================
// Core executeStep()
// ============================================================================

void TLASSO_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    // Set openMP dominance for the entire algorithm
    EigenSingleThreadGuard eigen_single_thread_guard;

    while (currentStep_ < maxSteps_ &&
           !inactives_.empty() &&
           actives_.size() < effective_n_ &&
           (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)) {

        // ========================================================
        // STEP 1: Find maximum correlation
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
        // STEP 4: Equi-angular computation
        // ========================================================
        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);
        Eigen::VectorXd u = equiangularVector(w_A);

        // ========================================================
        // STEP 5: Compute LARS step size and projection
        // ========================================================
        auto [gamma_entry, a] = computeStepSize(Cmax, u);

        // ========================================================
        // STEP 6: T-LASSO modification - check for sign changes
        // ========================================================
        std::vector<bool> drops(actives_.size(), false);
        double gamma = computeGammaSignChange(gamma_entry, drops, w_A);

        // ========================================================
        // STEP 7: Update beta coefficients and residuals
        // ========================================================
        updateBetaPath(w_A, gamma);
        r_ -= gamma * u;

        // ========================================================
        // STEP 8: Update correlations
        // ========================================================
        updateCorrelations(gamma, a);

        // ========================================================
        // STEP 9: T-LASSO drops and update inactives internally
        // ========================================================
        processLassoDrops(drops);
        // Record after the drops so this step's snapshot has the dropped
        // coefficients at exactly zero (absent from the support).
        recordBetaStep();

        // ========================================================
        // STEP 10: Update diagnostics and tracking
        // ========================================================
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 11: Update inactive set for next iteration (collinerity)
        // ========================================================
        updateInactiveSet();
    }
}


// ============================================================================
// Helper Implementations
// ============================================================================






// ============================================================================
// Serialization
// ============================================================================

void TLASSO_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TLASSO_Solver TLASSO_Solver::load(const std::string& filename,
                                Eigen::Map<Eigen::MatrixXd>& X,
                                Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TLASSO_Solver>(filename, X, D);
}

// ============================================================================
}  // namespace trex::tsolvers::linear_model::lars_based
