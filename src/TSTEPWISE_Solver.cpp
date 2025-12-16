

#include "TSTEPWISE_Solver.hpp"

// ============================================================================
// Core Algorithm Implementation: executeStep()
// ============================================================================

void TSTEPWISE_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    struct StepTimings {
        double find_max_corr = 0.0;
        double update_active = 0.0;
        double equiangular = 0.0;
        double beta_update = 0.0;
        double residual_update = 0.0;
        double inactive_update = 0.0;
        double corr_update = 0.0;
        std::size_t num_steps = 0;
    } timings;

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < effective_n_ &&
        (count_active_dummies_ < T_stop || !early_stop)
    ) {

        // ========================================================
        // STEP 1: Use pre-computed correlations
        // ========================================================
        auto t1 = std::chrono::high_resolution_clock::now();
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        auto t2 = std::chrono::high_resolution_clock::now();
        timings.find_max_corr += std::chrono::duration<double, std::milli>(t2-t1).count();

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
        t1 = std::chrono::high_resolution_clock::now();
        actions_.emplace_back(updateActiveSet(new_vars));
        t2 = std::chrono::high_resolution_clock::now();
        timings.update_active += std::chrono::duration<double, std::milli>(t2-t1).count();

        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; aborting.");
            break;
        }

        // ========================================================
        // STEP 4: Full step to axis-aligned solution
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);
        Eigen::VectorXd u = equiangularVector(w_A);

        // full orthogonal step (reduces correlations to zero)
        double gamma = Cmax / A_A_;
        t2 = std::chrono::high_resolution_clock::now();
        timings.equiangular += std::chrono::duration<double, std::milli>(t2-t1).count();

        // ========================================================
        // STEP 5: Update beta coefficients
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateBetaPath(w_A, gamma);
        t2 = std::chrono::high_resolution_clock::now();
        timings.beta_update += std::chrono::duration<double, std::milli>(t2-t1).count();

        // ========================================================
        // STEP 6: Update residuals
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        r_ -= gamma * u;
        t2 = std::chrono::high_resolution_clock::now();
        timings.residual_update += std::chrono::duration<double, std::milli>(t2-t1).count();

        // ========================================================
        // STEP 7: Update diagnostics
        // ========================================================
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        // Track dummy count and compute DoF (includes dummies + intercept)
        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 8: Update inactive set
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateInactiveSet();
        t2 = std::chrono::high_resolution_clock::now();
        timings.inactive_update += std::chrono::duration<double, std::milli>(t2-t1).count();

        // ==================================================================
        // STEP 9: Recompute correlations for next iteration and reset signs
        // ==================================================================
        t1 = std::chrono::high_resolution_clock::now();
        zeroAllSigns();
        recomputeCorrelations();
        t2 = std::chrono::high_resolution_clock::now();
        timings.corr_update += std::chrono::duration<double, std::milli>(t2-t1).count();

        timings.num_steps++;
    }

    // ============================================================
    // Print profile summary
    // ============================================================

    if (verbose_ && timings.num_steps > 0) {
        std::cout << "\n=== T-STEPWISE Profile Summary ===\n";
        std::cout << "Total steps: " << timings.num_steps << "\n\n";

        std::cout << "Time per step (avg):\n" << std::fixed << std::setprecision(1);
        printProfileLine("Find max corr:", timings.find_max_corr / timings.num_steps, " ms");
        printProfileLine("Update active:", timings.update_active / timings.num_steps, " ms");
        printProfileLine("Equiangular dir:", timings.equiangular / timings.num_steps, " ms");
        printProfileLine("Update beta:", timings.beta_update / timings.num_steps, " ms");
        printProfileLine("Update residual:", timings.residual_update / timings.num_steps, " ms");
        printProfileLine("Update inactives:", timings.inactive_update / timings.num_steps, " ms");
        printProfileLine("Update correlations:", timings.corr_update / timings.num_steps, " ms");

        double total = timings.find_max_corr + timings.update_active +
                       timings.equiangular + timings.beta_update +
                       timings.residual_update + timings.inactive_update +
                       timings.corr_update;

        std::cout << "\nPercentage breakdown:\n";
        printProfileLine("Find max corr:", 100.0 * timings.find_max_corr / total, "%");
        printProfileLine("Update active:", 100.0 * timings.update_active / total, "%");
        printProfileLine("Equiangular dir:", 100.0 * timings.equiangular / total, "%");
        printProfileLine("Update beta:", 100.0 * timings.beta_update / total, "%");
        printProfileLine("Update residual:", 100.0 * timings.residual_update / total, "%");
        printProfileLine("Update inactives:", 100.0 * timings.inactive_update / total, "%");
        printProfileLine("Update correlations:", 100.0 * timings.corr_update / total, "%");
        std::cout << "====================================\n\n";
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
    // Full recomputation: X_inactive^T * r
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < inactives_.size(); ++i) {
        std::size_t j = inactives_[i];
        correlations_(j) = X_->col(j).dot(r_);
    }
}


// ============================================================================
// Serialization Implementation
// ============================================================================

TSTEPWISE_Solver TSTEPWISE_Solver::load(const std::string& filename,
                                        Eigen::Map<Eigen::MatrixXd>& X) {

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
