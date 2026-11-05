// ============================================================================
// tstagewise_solver.cpp
// ============================================================================
/**
 * @file tstagewise_solver.cpp
 *
 * @brief Implementation of TSTAGEWISE_Solver, a T-LARS-based stagewise
 * regression solver.
 * It can become mathematically equivalent to LARS.
 */
// ============================================================================

#include <fstream>
#include <stdexcept>
#include <tsolvers/ad/linear_model/lars_based/tstagewise_solver.hpp>

// ============================================================================

// Embed into namespace trex::tsolvers::ad::linear_model::lars_based
namespace trex::tsolvers::ad::linear_model::lars_based {

// ============================================================================
// Constructors
// ============================================================================

TSTAGEWISE_Solver::TSTAGEWISE_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    const SolverOptions& options
)
    : TLARS_Solver(X, D, y, options, SolverTypeLarsBased::TSTAGEWISE) {
}

// ============================================================================
// Core executeStep()
// ============================================================================

void TSTAGEWISE_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();
    EigenThreadBudgetGuard eigen_thread_budget_guard(n_threads_);

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < effective_n_ &&
        (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)
    ) {
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
        // STEP 2: Increment step and add to active set
        // ========================================================
        currentStep_++;
        lambda_.push_back(Cmax);
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
        // Step 3: T-Stagewise NNLS Direction Check Loop
        // ========================================================
        nnls_dropped_step_.clear();
        Eigen::VectorXd w_A = computeStagewiseDirection();

        if (actives_.empty()) {
            logWarning("Active set is empty after NNLS reduction; terminating.");
            // The NNLS removals are already recorded in this step's actions;
            // conclude the step consistently as an empty model: zero
            // coefficients, residual = y, and diagnostics for this step.
            clearRunningBeta();
            recordBetaStep();
            r_ = y_;
            RSS_.push_back(r_.dot(r_));
            R2_.push_back(0.0);
            updateDummyTracking();
            DoF_.push_back(interceptDoF());
            updateInactiveSet();
            break;
        }

        // ========================================================
        // Step 4: Update Inactives
        // ========================================================
        // If NNLS dropped variables, they must rejoin inactives_
        // before computeStepSize
        updateInactiveSet();

        // The variables removed in this step still sit at Cmax, so their step
        // roots are rounding noise (a noise root above eps gives a near-zero
        // step, re-entry and removal again). Their correlation falls faster
        // than the active level, so they cannot catch up within the step:
        // no candidates here, re-admitted with exact correlations after the move.
        std::vector<std::size_t> excluded;
        for (std::size_t v : nnls_dropped_step_) {
            if (std::erase(inactives_, v) > 0) { excluded.push_back(v); }
        }

        // ========================================================
        // Step 5: Equi-angular projection and Step Size
        // ========================================================
        Eigen::VectorXd u = equiangularVector(w_A);
        auto [gamma, a] = computeStepSize(Cmax, u);

        // ========================================================
        // Step 6: Update Beta Path and Residuals
        // ========================================================
        // NNLS removals (Step 3) leave their coefficients frozen in the
        // running support — only the active directions advance.
        updateBetaPath(w_A, gamma);
        recordBetaStep();
        r_ -= gamma * u;

        // ========================================================
        // Step 7: Update Correlations
        // ========================================================
        updateCorrelations(gamma, a);
        // Re-admit the variables removed in this step with exact correlations
        for (std::size_t v : excluded) {
            inactives_.push_back(v);
            refreshDroppedCorrelation(v);
        }

        // ========================================================
        // Step 8: Update diagnostics and dummy tracking
        // ========================================================
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        updateDummyTracking();
        DoF_.push_back(actives_.size() + interceptDoF());
    }
}

// ============================================================================
// Stagewise Direction Computation
// ============================================================================

Eigen::VectorXd TSTAGEWISE_Solver::computeStagewiseDirection() {
    bool directions_valid = false;
    Eigen::VectorXd w_A;

    while (!directions_valid && !actives_.empty()) {
        Eigen::VectorXd sign_vec = signVector();

        // Compute unscaled direction
        Eigen::VectorXd Gi1 = backsolve(R_, backsolveT(R_, sign_vec));

        // Check direction-sign consistency
        Eigen::VectorXd directions = Gi1.array() * sign_vec.array();

        if ((directions.array() > eps_).all()) {
            // All variables move in the correct direction
            directions_valid = true;

            // Scale to equiangular direction
            double A = 1.0 / std::sqrt((Gi1.array() * sign_vec.array()).sum());

            // Compute final direction vector for active set
            w_A = A * Gi1;

            // Store step size computation
            A_A_ = A;

        } else {
            // Monotonicity violation: perform NNLS to find positive active subset and
            // re-iterate
            processStagewiseNNLS(directions);
        }
    }
    return w_A;
}

// ============================================================================
// NNLS Subroutine
// ============================================================================

void TSTAGEWISE_Solver::processStagewiseNNLS(const Eigen::Ref<const Eigen::VectorXd>& directions) {

    // 1. Find most violating variable (minimum direction <= 0)
    Eigen::Index drop_idx = 0;
    directions.minCoeff(&drop_idx);

    std::size_t dropped_var = actives_[drop_idx];

    // 2. Update dummy tracking: a dummy removed with a frozen nonzero
    // coefficient stays in the support and keeps counting toward T; one
    // removed before any move (coefficient zero) leaves the count
    if (dropped_var >= dummy_start_idx_ && count_active_dummies_ > 0) {
        if (runningBeta(dropped_var) == 0.0) {
            count_active_dummies_--;
            logInfo(concatMsg("Dummy variable ", dropped_var,
                              " removed during NNLS reduction (count: ",
                              count_active_dummies_, ")\n"));
        } else {
            logInfo(concatMsg("Dummy variable ", dropped_var,
                              " removed during NNLS reduction, frozen coefficient keeps counting (count: ",
                              count_active_dummies_, ")\n"));
        }
    }

    // 3. Downdate Cholesky factor R_
    R_ = downdateR(R_, drop_idx, eps_);

    // 4. Record the removal action in current step
    // Negative index indicates removal
    actions_.back().push_back(actionDrop(dropped_var));

    // 5. Refresh correlation for the dropped variable for potential re-entry
    // (its correlations_ entry is stale from when it entered the active set)
    refreshDroppedCorrelation(dropped_var);

    // 6. Remove from active set and signs
    actives_.erase(actives_.begin() + drop_idx);
    Sign_.erase(Sign_.begin() + drop_idx);

    // 7. Mark for inactive set rebuild at the end of the step
    any_dropped_ = true;
    num_removals_++;
    nnls_dropped_step_.push_back(dropped_var);
}

// ============================================================================
// Internal Helpers & Serialization
// ============================================================================


// ============================================================================
// De-/Serialization
// ============================================================================

void TSTAGEWISE_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }


TSTAGEWISE_Solver TSTAGEWISE_Solver::load(const std::string& filename,
                                          Eigen::Map<Eigen::MatrixXd>& X,
                                          Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TSTAGEWISE_Solver>(filename, X, D);
}

// ============================================================================
} /* End of namespace trex::tsolvers::ad::linear_model::lars_based */
