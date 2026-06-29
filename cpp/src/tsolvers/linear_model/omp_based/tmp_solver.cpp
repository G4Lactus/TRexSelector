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
    Eigen::setNbThreads(1);

    while (currentStep_ < maxSteps_ &&
           actives_.size() < effective_n_ &&
           (count_active_dummies_ < T_stop || !early_stop)) {

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
    Eigen::Index p_tot = static_cast<Eigen::Index>(p_original_ + num_dummies_);

    if (currentStep_ >= static_cast<std::size_t>(betaPath_.cols())) {
        betaPath_.conservativeResize(p_tot,
                                     static_cast<Eigen::Index>(currentStep_ + 1));
    }

    Eigen::VectorXd beta_hat = Eigen::VectorXd::Zero(p_tot);

    std::size_t k = actives_.size();
    for (std::size_t i = 0; i < k; ++i) {
        beta_hat[static_cast<Eigen::Index>(actives_[i])] =
            active_coefficients_[static_cast<Eigen::Index>(i)];
    }

    betaPath_.col(static_cast<Eigen::Index>(currentStep_)) = beta_hat;
}

// ==================================================================================
// Overrides
// ==================================================================================

std::pair<double, std::vector<std::size_t>>
TMP_Solver::findTiedMaxCorrelations() const {
    double Cmax = 0.0;
    std::vector<std::size_t> tied;
    tied.reserve(10);
    std::size_t p = p_original_ + num_dummies_;

    std::unordered_set<std::size_t> dropped_set;
    if (!dropped_indices_.empty()) {
        dropped_set = std::unordered_set<std::size_t>(dropped_indices_.begin(),
                                                       dropped_indices_.end());
    }

    for (std::size_t j = 0; j < p; ++j) {
        if (dropped_set.find(j) != dropped_set.end()) continue;

        double abs_corr = std::abs(correlations_[static_cast<Eigen::Index>(j)]);

        if (abs_corr > Cmax + eps_) {
            Cmax = abs_corr;
            tied.clear();
            tied.push_back(j);
        } else if (abs_corr >= Cmax - eps_) {
            tied.push_back(j);
        }
    }

    return {Cmax, tied};
}

std::vector<int>
TMP_Solver::updateActiveSet(const std::vector<std::size_t>& new_vars) {

    std::vector<int> actions_this_step;

    for (auto j_new : new_vars) {
        bool already_active = (std::find(actives_.begin(),
                                         actives_.end(), j_new)
                                != actives_.end());

        actions_this_step.push_back(static_cast<int>(j_new));

        if (!already_active) {
            // New atom: append to active set, initialize coefficient to 0
            actives_.push_back(j_new);
            num_additions_++;

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
    std::size_t p           = p_original_ + num_dummies_;
    std::size_t num_dropped = dropped_indices_.size();

    if (num_dropped == 0) {
        #pragma omp parallel for schedule(static)
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p); ++j) {
            correlations_[j] = getColumn(j).dot(r_);
        }
        return;
    }

    if (num_dropped > 500) {
        std::unordered_set<std::size_t> dropped_set(dropped_indices_.begin(),
                                                     dropped_indices_.end());
        #pragma omp parallel for schedule(static)
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p); ++j) {
            correlations_[j] = dropped_set.contains(static_cast<std::size_t>(j))
                                ? 0.0
                                : getColumn(j).dot(r_);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p); ++j) {
            bool is_dropped = false;
            for (auto idx : dropped_indices_) {
                if (idx == static_cast<std::size_t>(j)) { is_dropped = true; break; }
            }
            correlations_[j] = is_dropped ? 0.0 : getColumn(j).dot(r_);
        }
    }
}

// ==================================================================================
// De-/Serialization
// ==================================================================================

void TMP_Solver::save(const std::string& filename) const {

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error(
            concatMsg(solverTypeToString(), "::save: Cannot open '", filename, "'")
        );
    }

    cereal::PortableBinaryOutputArchive oarchive(ofs);
    oarchive(*this);
    logInfo(concatMsg("[", solverTypeToString(), "] saved to '", filename, "'\n"));
}

TMP_Solver TMP_Solver::load(
    const std::string& filename,
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D
) {
    TMP_Solver tmp;
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error(
            tmp.concatMsg(tmp.solverTypeToString(),
                          "::load: Cannot open '", filename, "'")
        );
    }

    cereal::PortableBinaryInputArchive iarchive(is);
    iarchive(tmp);

    tmp.reconnect(X, D);
    return tmp;
}

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
