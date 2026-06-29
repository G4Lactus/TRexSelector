// ==================================================================================
// tgp_solver.cpp
// ==================================================================================

// std includes
#include <fstream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/omp_based/tgp_solver.hpp>

// ==================================================================================

// Embedded into trex::tsolvers::linear_model::omp_based namespace
namespace trex::tsolvers::linear_model::omp_based {


// ==================================================================================
// Constructors
// ==================================================================================

TGP_Solver::TGP_Solver(SolverTypeOMPBased solver_type)
    : TOMP_Solver(solver_type) {}

// ==================================================================================
// Main Algorithm: executeStep()
// ==================================================================================

void TGP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();
    // Set openMP dominance for the entire algorithm
    Eigen::setNbThreads(1);

    while (currentStep_ < maxSteps_ &&
           actives_.size() < effective_n_ &&
           (count_active_dummies_ < T_stop || !early_stop)) {

        // =============================================================
        // STEP 1: Find max abs correlation |c^{n}| among ALL variables
        // =============================================================

        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0 or no variables to add; exiting.");
            break;
        }

        pruneTiedDummies(new_vars, T_stop, early_stop);

        // =============================================================
        // STEP 2: Update active set (with inherited dummy tracking)
        // =============================================================

        actions_.emplace_back(updateActiveSet(new_vars));
        if (actives_.empty()) {
            logWarning("Active set is empty; aborting.");
            break;
        }

        // ===========================================================
        // STEP 3: Compute gradient direction: d^{n} = c^{n}_A
        // ===========================================================

        computeGradientDirection();

        // ===========================================================
        // STEP 4: Projected direction: p^{n} = X_A * d^{n}
        // ===========================================================
        projectDirection();

        // ===========================================================
        // STEP 5: Compute step size
        // alpha^{n} = <r^{n-1}, p^{n}> / ||p^{n}||^2
        // ===========================================================

        step_size_ = computeStepSize();  // alpha_n = <r, c> / ||c||^2
        if (std::abs(step_size_) < eps_) {
            logInfo("Step size approx 0; exiting.");
            break;
        }

        // ==========================================================
        // STEP 6: Update active coefficients
        // beta^{n}_A = beta^{n-1}_A + alpha^{n} * d^{n}
        // ==========================================================

        updateActiveCoefficients();

        // ==========================================================
        // STEP 7: Update residuals:
        // r^{n} = r^{n-1} - alpha^{n} * p^{n}
        // ==========================================================

        updateResiduals();

        // ==========================================================
        // STEP 8: Update path and diagnostics
        // ==========================================================

        currentStep_++;
        updateBetaPath();

        double rss = r_.dot(r_);
        RSS_.emplace_back(rss);
        R2_.emplace_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();
        DoF_.emplace_back(actives_.size() + (intercept_ ? 1 : 0));

        // ==========================================================
        // STEP 9: Update correlations for ALL variables
        // c^{n} = X^{T} * r
        // ==========================================================

        updateCorrelations();
    }
}


// ==================================================================================
// GP-Specific Helper Methods
// ==================================================================================

void TGP_Solver::computeGradientDirection() {

    // Gradient for active variables: d^{n} = c^{n}_A = X^{T}_A r
    Eigen::Index k = static_cast<Eigen::Index>(actives_.size());
    direction_.resize(k);

    #pragma omp parallel for schedule(static) if (k > 100)
    for (Eigen::Index i = 0; i < k; ++i) {
        direction_[i] = correlations_[static_cast<Eigen::Index>(actives_[i])];
    }
}


void TGP_Solver::projectDirection() {

    // Projected direction into data space: p^{n} = X_A * d^{n}
    Eigen::Index n = X_->rows();
    projected_direction_.resize(n);
    projected_direction_.setZero();

    // p^{n} = X_A * d^{n} (manual loop for memory efficiency in p >> n)
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(actives_.size()); ++i) {
        projected_direction_ += direction_[i] *
                                getColumn(static_cast<Eigen::Index>(actives_[i]));
    }
}


double TGP_Solver::computeStepSize() {
    // Step size: alpha^{n} = <r^{n-1}, p^{n}> / ||p^{n}||^2
    double numerator = r_.dot(projected_direction_);
    double denominator = projected_direction_.squaredNorm();

    if (denominator < eps_) { // implicit collinearity check
        logWarning("||p^{n}||^2 ~0; numerical issue.");
        return 0.0; // no update
    }

    double step = numerator / denominator;

    // Sanity check for huge step
    if (std::abs(step) > 1e10) {
        logWarning("Computed step size is huge; numerical issue.");
        return std::copysign(1e10, step); // cap and preserve sign
    }

    return step;
}


void TGP_Solver::updateActiveCoefficients() {
    std::size_t k = actives_.size();

    // Ensure active_coefficients_ has correct size
    if (static_cast<std::size_t>(active_coefficients_.size()) != k) {
        active_coefficients_.conservativeResize(static_cast<Eigen::Index>(k));
    }

    // beta_A^n = beta_A^{n-1} + alpha^n * d^n
    #pragma omp parallel for schedule(static) if (k > 100)
    for (std::size_t i = 0; i < k; ++i) {
        active_coefficients_[static_cast<Eigen::Index>(i)] +=
            step_size_ * direction_[static_cast<Eigen::Index>(i)];
    }
}


void TGP_Solver::updateResiduals() {
    // GP residual update: r^{n} = r^{n-1} - alpha^{n} * p^{n}
    r_.noalias() -= step_size_ * projected_direction_;
}


void TGP_Solver::updateBetaPath() {
    Eigen::Index p_tot = static_cast<Eigen::Index>(p_original_ + num_dummies_);

    if (currentStep_ >= static_cast<std::size_t>(betaPath_.cols())) {
        betaPath_.conservativeResize(p_tot,
                                     static_cast<Eigen::Index>(currentStep_ + 1));
    }

    Eigen::VectorXd beta_hat = Eigen::VectorXd::Zero(p_tot);

    std::size_t k = actives_.size();
    #pragma omp parallel for schedule(static) if (k > 100)
    for (std::size_t i = 0; i < k; ++i) {
        beta_hat[static_cast<Eigen::Index>(actives_[i])] =
            active_coefficients_[static_cast<Eigen::Index>(i)];
    }
    betaPath_.col(static_cast<Eigen::Index>(currentStep_)) = beta_hat;
}


// ==================================================================================
// Overrides
// ==================================================================================

std::pair<double, std::vector<std::size_t>> TGP_Solver::findTiedMaxCorrelations() const {
    double Cmax = 0.0;
    std::vector<std::size_t> tied;
    tied.reserve(10);
    std::size_t p = p_original_ + num_dummies_;

    // Conditional set construction
    std::unordered_set<std::size_t> dropped_set;
    if (!dropped_indices_.empty()) {
        dropped_set = std::unordered_set<std::size_t>(dropped_indices_.begin(),
                                                      dropped_indices_.end());
    }

    for (std::size_t j = 0; j < p; ++j) {
        if (dropped_set.find(j) != dropped_set.end()) {
            continue; // skip dropped
        }

        double abs_corr = std::abs(correlations_[static_cast<Eigen::Index>(j)]);

        if (abs_corr > Cmax + eps_) {
                Cmax = abs_corr;
                tied.clear();
                tied.push_back(j);
        } else if (abs_corr >= Cmax - eps_) {
                // Tied at maximum
                tied.push_back(j);
        }
    }
    return {Cmax, tied};
}


std::vector<int> TGP_Solver::updateActiveSet(const std::vector<std::size_t>& new_vars) {

    std::vector<int> actions_this_step;

    for (auto j_new : new_vars) {
        // Check if variable is already active (GP re-selection)
        bool already_active = (std::find(actives_.begin(),
                               actives_.end(), j_new)
                               != actives_.end());

        actions_this_step.push_back(static_cast<int>(j_new));

        if (!already_active) {
            // New atom: add directly
            actives_.push_back(j_new);
            num_additions_++;

            // Initialize coefficient
            active_coefficients_.conservativeResize(
                static_cast<Eigen::Index>(actives_.size()));
            active_coefficients_(static_cast<Eigen::Index>(actives_.size() - 1)) = 0.0;

            // Track dummy entry
            if (j_new >= dummy_start_idx_) {
                count_active_dummies_++;
            }
        }
    }

    return actions_this_step;
}


void TGP_Solver::updateCorrelations() {
    // Update correlations  c^{n} = X^T r^{n} for ALL variables (not just inactives) to
    // allow re-selection in TGP
    std::size_t p = p_original_ + num_dummies_;
    std::size_t num_dropped = dropped_indices_.size();

    if (num_dropped == 0) {
        #pragma omp parallel for schedule(static)
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p); ++j) {
            correlations_[j] = getColumn(j).dot(r_);
        }
        return;

    } else if (num_dropped > 500) {
        std::unordered_set<std::size_t> dropped_set(
            dropped_indices_.begin(), dropped_indices_.end());
        #pragma omp parallel for schedule(static)
        for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(p); ++j) {
            correlations_[j] = dropped_set.contains(
                static_cast<std::size_t>(j)) ? 0.0 : getColumn(j).dot(r_);
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

void TGP_Solver::save(const std::string& filename) const {

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


TGP_Solver TGP_Solver::load(
    const std::string& filename,
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D
) {

    TGP_Solver tgp;
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error(
            tgp.concatMsg(tgp.solverTypeToString(), "::load: Cannot open '", filename, "'")
        );
    }

    cereal::PortableBinaryInputArchive iarchive(is);
    iarchive(tgp);

    tgp.reconnect(X, D);
    return tgp;
}

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
