// ==================================================================================
// tmp_solver.cpp
// ==================================================================================
/**
* @file tmp_solver.cpp
*
* @brief Implementation of the Terminating Matching Pursuit (T-MP) solver.
*/
// ==================================================================================

// std includes
#include <algorithm>
#include <fstream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/omp_based/tmp_solver.hpp>

// ==================================================================================

// Embedded into trex::tsolvers::linear_model::omp_based namespace
namespace trex::tsolvers::linear_model::omp_based {

// ==================================================================================
// Constructors
// ==================================================================================

TMP_Solver::TMP_Solver(SolverTypeOMPBased solver_type)
    : TOMP_Solver(solver_type) {}

// ==================================================================================
// Main Algorithm: executeStep()
// ==================================================================================

void TMP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();
    EigenSingleThreadGuard eigen_single_thread_guard;

    while (currentStep_ < maxSteps_ &&
           actives_.size() < effective_n_ &&
           (!early_stop || T_stop == 0 || count_active_dummies_ < T_stop)) {

        // ==========================================================
        // STEP 1: Find max |c^{n}| among ALL variables
        // ==========================================================
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0 or no variables to add; exiting.");
            break;
        }

        pruneTiedDummies(new_vars, T_stop, early_stop);

        // MP: single-atom update — take primary atom after tie-breaking
        const std::size_t col_idx = new_vars[0];

        // ==========================================================
        // STEP 2: Update active set (with inherited dummy tracking)
        // ==========================================================
        actions_.emplace_back(updateActiveSet({col_idx}));
        if (actives_.empty()) {
            logWarning("Active set is empty; aborting.");
            break;
        }

        // ==========================================================
        // STEP 3: Step size
        // alpha^{n} = c^{n}_{i^{n}} = <r^{n-1}, x_{i^{n}}>
        // ==========================================================
        step_size_ = correlations_[static_cast<Eigen::Index>(col_idx)];
        if (std::abs(step_size_) < eps_) {
            logInfo("Step size approx 0; exiting.");
            break;
        }

        // ==========================================================
        // STEP 4: Increment coefficient: beta_A^{n}[i^{n}] += alpha^{n}
        // ==========================================================
        auto it = std::find(
            actives_.begin(), actives_.end(), col_idx);
        std::size_t atom_idx = static_cast<std::size_t>(
            std::distance(actives_.begin(), it));
        incrementCoefficient(atom_idx);

        // ==========================================================
        // STEP 5: Subtract atom contribution from residual:
        // r^{n} = r^{n-1} - alpha^{n} * x_{i^{n}}
        // ==========================================================
        subtractAtom(col_idx);

        // ==========================================================
        // STEP 6: Path and diagnostics
        // ==========================================================
        currentStep_++;
        updateBetaPath();

        double rss = r_.dot(r_);
        RSS_.emplace_back(rss);
        R2_.emplace_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();
        DoF_.emplace_back(actives_.size() + (intercept_ ? 1 : 0));

        // ==========================================================
        // STEP 7: Update correlations c^{n} = X^T r for ALL variables
        // ==========================================================
        updateCorrelations();
    }
}

// ==================================================================================
// MP-Specific Helper Methods
// ==================================================================================

void TMP_Solver::incrementCoefficient(std::size_t atom_idx) {
    // beta_A^{n}[i^{n}] = beta_A^{n-1}[i^{n}] + alpha^{n}
    active_coefficients_[static_cast<Eigen::Index>(atom_idx)] += step_size_;
}

void TMP_Solver::subtractAtom(std::size_t col_idx) {
    // r^{n} = r^{n-1} - alpha^{n} * x_{i^{n}}
    r_.noalias() -= step_size_ * getColumn(col_idx);
}

void TMP_Solver::updateBetaPath() {
    // active_coefficients_ is already the sparse coefficient vector over
    // actives_ — snapshot it directly.
    setRunningBeta(actives_, active_coefficients_);
    recordBetaStep();
}

// ==================================================================================
// Overrides
// ==================================================================================

std::pair<double, std::vector<std::size_t>>
TMP_Solver::findTiedMaxCorrelations() const {
    // Scan ALL non-dropped variables (atom re-selection is allowed in MP)
    return findTiedMaxCorrelationsAllNonDropped();
}

std::vector<int>
TMP_Solver::updateActiveSet(const std::vector<std::size_t>& new_vars) {

    std::vector<int> actions_this_step;

    for (auto j_new : new_vars) {
        bool already_active = (std::find(actives_.begin(),
                                         actives_.end(), j_new)
                                != actives_.end());

        actions_this_step.push_back(actionAdd(j_new));

        if (!already_active) {
            // New atom: append to active set, initialize coefficient to 0
            actives_.push_back(j_new);
            num_additions_++;

            // Keep the actives/inactives partition invariant intact for the
            // public getters (selection itself scans all non-dropped columns)
            std::erase(inactives_, j_new);

            active_coefficients_.conservativeResize(
                static_cast<Eigen::Index>(actives_.size()));
            active_coefficients_(
                static_cast<Eigen::Index>(actives_.size() - 1)) = 0.0;

            if (j_new >= dummy_start_idx_) {
                count_active_dummies_++;
            }
        }
        // Re-selected atom: action recorded, no structural change to actives_
        // Coefficient updated via beta_A^n[i^n] += alpha^n
    }

    return actions_this_step;
}

void TMP_Solver::updateCorrelations() {
    // c^n = X^T r for ALL non-dropped variables (to allow re-selection)
    refreshAllCorrelations();
}

// ==================================================================================
// De-/Serialization
// ==================================================================================

void TMP_Solver::save(const std::string& filename) const { saveImpl(*this, filename); }

TMP_Solver TMP_Solver::load(const std::string& filename,
                            Eigen::Map<Eigen::MatrixXd>& X,
                            Eigen::Map<Eigen::MatrixXd>& D) {
    return loadImpl<TMP_Solver>(filename, X, D);
}

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
