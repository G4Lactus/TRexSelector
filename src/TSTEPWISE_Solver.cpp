
#include "TSTEPWISE_Solver.hpp"

// ============================================================================
// Core Algorithm Implementation: executeStep()
// ============================================================================

void TSTEPWISE_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < effective_n_ &&
        (count_active_dummies_ < T_stop || !early_stop)
    ) {

        // ========================================================
        // STEP 1: Use pre-computed correlations
        // ========================================================

        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty()) {
            logInfo("No variables to add; exiting.");
            break;
        }

        if (Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0; exiting.");
            break;
        }

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
        // STEP 4: Full step to axis-aligned solution
        // ========================================================

        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);
        Eigen::VectorXd u = equiangularVector(w_A);
        double gamma = Cmax / A_A_;

        // ========================================================
        // STEP 5: Update beta coefficients
        // ========================================================

        updateBetaPath(w_A, gamma);

        // ========================================================
        // STEP 6: Update residuals
        // ========================================================

        r_ -= gamma * u;

        // ========================================================
        // STEP 7: Update diagnostics
        // ========================================================

        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        // T-LARS: Track dummy count and compute DoF (includes dummies)
        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 8: Update inactive set
        // ========================================================

        updateInactiveSet();

        // ==================================================================
        // STEP 9: Recompute correlations for next iteration and reset signs
        // ==================================================================

        zeroAllSigns();
        computeCorrelations();
    }
}


// ============================================================================
// Internal Helper
// ============================================================================

void TSTEPWISE_Solver::zeroAllSigns() {
    // Reset signs to zero for orthogonal (axis-aligned) steps
    Sign_.assign(Sign_.size(), 0);
}



// ============================================================================
// Serialization Implementation
// ============================================================================

TSTEPWISE_Solver TSTEPWISE_Solver::load(const std::string& filename, Eigen::Map<Eigen::MatrixXd>& X) {

    TSTEPWISE_Solver tstepwise;  // Uses no-args constructor

    const std::string solver{TSTEPWISE_Solver::solverTypeToString(SolverType::TStepwise)};

    // 1. Deserialize from file with error handling
    {
        std::ifstream is(filename, std::ios::binary);
        if (!is.is_open()) {
            std::ostringstream oss;
            oss << solver << "::load: Can't open file " << filename;
            throw std::runtime_error(oss.str());
        }

        try {
            cereal::PortableBinaryInputArchive iarchive(is);
            iarchive(tstepwise);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << solver << "::load: Deserialization failed - " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    // 2. Validate loaded state integrity
    if (tstepwise.betaPath_.rows() == 0) {
        throw std::runtime_error(
            tstepwise.concatMsg(
                solver, "::load: Loaded state is corrupted (betaPath_ is empty)"
            )
        );
    }

    // 3. Reconnect with error handling
    try {
        tstepwise.reconnect(X);

        tstepwise.logInfo(tstepwise.concatMsg(
            "[", tstepwise.solverTypeToString(), "] loaded from '", filename, "'\n",
            "  - Algorithm: ", tstepwise.solverTypeToString(), "\n",
            "  - Loaded at step: ", tstepwise.currentStep_, "/", tstepwise.maxSteps_, "\n",
            "  - Active variables: ", tstepwise.actives_.size(), "\n",
            "  - Matrix dimensions: ", X.rows(), " x ", X.cols(), "\n",
            "  - Status: Ready for continued execution"
        ));
    } catch (const std::exception& e) {
        throw std::runtime_error(
            tstepwise.concatMsg(
                tstepwise.solverTypeToString(), "::load: Reconnection failed - ", e.what()
            )
        );
    }

    return tstepwise;
}
