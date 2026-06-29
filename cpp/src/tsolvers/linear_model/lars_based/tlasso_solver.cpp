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
    Eigen::setNbThreads(1);

    while (currentStep_ < maxSteps_ &&
           !inactives_.empty() &&
           actives_.size() < effective_n_ &&
           (count_active_dummies_ < T_stop || !early_stop)) {

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

double TLASSO_Solver::getCyclingRatio() const {
    if (num_additions_ == 0) { return 0.0; }

    return static_cast<double>(num_removals_) / static_cast<double>(num_additions_);
}


double TLASSO_Solver::computeGammaSignChange(double gamhat,
                                             std::vector<bool>& drops,
                                             const
                                             Eigen::Ref<const Eigen::VectorXd>& w_A) const {

    // Extract current beta values for active variables
    Eigen::VectorXd beta_curr(actives_.size());
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(actives_.size()); ++i) {
        beta_curr(i) = betaPath_(static_cast<Eigen::Index>(actives_[i]),
                                        static_cast<Eigen::Index>(currentStep_ - 1));
    }

    // Find zero crossing:  z = -β / w  (step size where β + γw = 0)
    Eigen::VectorXd z = -beta_curr.array() / w_A.array();

    // Find minimum positive z (first zero-crossing)
    double zmin = std::numeric_limits<double>::max();
    bool found_crossing = false;

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(actives_.size()); ++i) {
        if (z(i) > eps_ && z(i) < zmin) {
            zmin = z(i);
            found_crossing = true;
        }
    }

    if (found_crossing && zmin < gamhat) {
        // Coefficient will cross zero before next variable enters
        gamhat = zmin;

        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(actives_.size()); ++i) {
            drops[i] = (std::abs(z(i) - zmin) < eps_);
        }
    }

    return gamhat;
}


void TLASSO_Solver::processLassoDrops(std::vector<bool>& drops) {

    // Process drops in reverse order to maintain indices
    for (std::size_t i = actives_.size(); i-- > 0;) {
        if (drops[i]) {
            std::size_t dropped_var = actives_[i];

            // Track if dropping a dummy
            if (dropped_var >= dummy_start_idx_ && count_active_dummies_ > 0) {
                count_active_dummies_--;
                logInfo(concatMsg("Dummy variable ", dropped_var,
                                  " removed from active set (count: ",
                                  count_active_dummies_, ")"));
            }

            // Downdate Cholesky factor
            R_ = downdateR(R_, i, eps_);

            // Zero out coefficient
            betaPath_(static_cast<Eigen::Index>(dropped_var),
                      static_cast<Eigen::Index>(currentStep_)) = 0.0;

            // Record negative action (removal)
            actions_.back().push_back(-static_cast<int>(dropped_var));

            // Re-add to inactives_ for cycling support
            inactives_.push_back(dropped_var);

            // Compute correlation for dropped variable for potential re-entry
            Eigen::Index j = static_cast<Eigen::Index>(dropped_var);
            correlations_(j) = getColumn(dropped_var).dot(r_);
            correlation_compensation_(j) = 0.0;

            // Remove from active set
            actives_.erase(actives_.begin() + static_cast<Eigen::Index>(i));
            Sign_.erase(Sign_.begin() + static_cast<Eigen::Index>(i));

            // Increment removal counter
            num_removals_++;
        }
    }
}


// ============================================================================
// Serialization
// ============================================================================

void TLASSO_Solver::save(const std::string& filename) const {

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


TLASSO_Solver TLASSO_Solver::load(const std::string& filename,
                                  Eigen::Map<Eigen::MatrixXd>& X,
                                  Eigen::Map<Eigen::MatrixXd>& D) {

    TLASSO_Solver tlasso;
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error(
            tlasso.concatMsg(tlasso.solverTypeToString(), "::load: Cannot open '", filename, "'")
        );
    }

    cereal::PortableBinaryInputArchive iarchive(is);
    iarchive(tlasso);

    tlasso.reconnect(X, D);

    return tlasso;
}

// ============================================================================
}  // namespace trex::tsolvers::linear_model::lars_based
