
#include "TLASSO_Solver.hpp"


// ============================================================================
// Core executeStep() Implementation
// ============================================================================

void TLASSO_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    // Add profiling structure
    struct StepTimings {
        double find_max_corr = 0.0;
        double update_active = 0.0;
        double equiangular = 0.0;
        double step_size = 0.0;
        double gamma_sign = 0.0;
        double beta_update = 0.0;
        double residual_update = 0.0;
        double lasso_drops = 0.0;
        double inactive_update = 0.0;
        double corr_update = 0.0;
        std::size_t num_steps = 0;
    } timings;

    while (currentStep_ < maxSteps_ &&
           !inactives_.empty() &&
           actives_.size() < effective_n_ &&
           (count_active_dummies_ < T_stop || !early_stop)) {

        // ========================================================
        // STEP 1: Find maximum correlation
        // ========================================================
        auto t1 = std::chrono::high_resolution_clock::now();
        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        auto t2 = std::chrono::high_resolution_clock::now();
        timings.find_max_corr += std::chrono::duration<double, std::milli>(t2 - t1).count();

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
        // STEP 3: Active set update (with inherited dummy tracking)
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        actions_.emplace_back(updateActiveSet(new_vars));
        t2 = std::chrono::high_resolution_clock::now();
        timings.update_active += std::chrono::duration<double, std::milli>(t2 - t1).count();

        if (actives_.empty() || R_.size() == 0) {
            logWarning("Active set or Cholesky is empty; aborting.");
            break;
        }

        // ========================================================
        // STEP 4: Equi-angular computation
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd sign_vec = signVector();
        Eigen::VectorXd w_A = equiangularDirection(sign_vec);
        Eigen::VectorXd u = equiangularVector(w_A);
        t2 = std::chrono::high_resolution_clock::now();
        timings.equiangular += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // ========================================================
        // STEP 5: T-LASSO modification - check for sign changes
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        auto [gamma_entry, a] = computeStepSize(Cmax, u);
        t2 = std::chrono::high_resolution_clock::now();
        timings.step_size += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // ========================================================
        // STEP 6: T-LASSO modification - check for sign changes
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        std::vector<bool> drops(actives_.size(), false);
        double gamma = computeGammaSignChange(gamma_entry, drops, w_A);
        t2 = std::chrono::high_resolution_clock::now();
        timings.gamma_sign += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // ========================================================
        // STEP 7: Update beta coefficients
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateBetaPath(w_A, gamma);
        t2 = std::chrono::high_resolution_clock::now();
        timings.beta_update += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // ========================================================
        // STEP 8: Update residuals
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        r_ -= gamma * u;
        t2 = std::chrono::high_resolution_clock::now();
        timings.residual_update += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // ========================================================
        // STEP 9: Update diagnostics
        // ========================================================
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        // ========================================================
        // STEP 10: Handle T-LASSO drops (before dummy tracking)
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        processLassoDrops(drops);
        t2 = std::chrono::high_resolution_clock::now();
        timings.lasso_drops += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Record dummy count and DoF AFTER all adds/removes
        dummies_at_step_.push_back(count_active_dummies_);
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // STEP 11: Update inactive set
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateInactiveSet();
        t2 = std::chrono::high_resolution_clock::now();
        timings.inactive_update += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // ========================================================
        // STEP 12: Recompute correlations for next iteration
        // ========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateCorrelations(gamma, a);
        t2 = std::chrono::high_resolution_clock::now();
        timings.corr_update += std::chrono::duration<double, std::milli>(t2 - t1).count();

        timings.num_steps++;
    }

    // ============================================================
    // Print profiling summary
    // ============================================================
    if (verbose_ && timings.num_steps > 0) {
        std::cout << "\n=== T-LASSO Profiling Summary ===\n";
        std::cout << "Total steps: " << timings.num_steps << "\n";

        std::cout << "Time per step (avg):\n" << std::fixed << std::setprecision(1);
        printProfileLine("Find max corr:", timings.find_max_corr / timings.num_steps, " ms");
        printProfileLine("Update active:", timings.update_active / timings.num_steps, " ms");
        printProfileLine("Equiangular dir:", timings.equiangular / timings.num_steps, " ms");
        printProfileLine("Compute stepsize:", timings.step_size / timings.num_steps, " ms");
        printProfileLine("Gamma sign check:", timings.gamma_sign / timings.num_steps, " ms");
        printProfileLine("Update beta:", timings.beta_update / timings.num_steps, " ms");
        printProfileLine("Update residual:", timings.residual_update / timings.num_steps, " ms");
        printProfileLine("LASSO drops:", timings.lasso_drops / timings.num_steps, " ms");
        printProfileLine("Update inactives:", timings.inactive_update / timings.num_steps, " ms");
        printProfileLine("Update correlations:", timings.corr_update / timings.num_steps, " ms");

        double total = timings.find_max_corr + timings.update_active +
                       timings.equiangular + timings.step_size +
                       timings.gamma_sign + timings.beta_update +
                       timings.residual_update + timings.lasso_drops +
                       timings.inactive_update + timings.corr_update;

        std::cout << "\nPercentage breakdown:\n";
        printProfileLine("Find max corr:", 100.0 * timings.find_max_corr / total, "%");
        printProfileLine("Update active:", 100.0 * timings.update_active / total, "%");
        printProfileLine("Equiangular dir:", 100.0 * timings.equiangular / total, "%");
        printProfileLine("Compute stepsize:", 100.0 * timings.step_size / total, "%");
        printProfileLine("Gamma sign check:", 100.0 * timings.gamma_sign / total, "%");
        printProfileLine("Update beta:", 100.0 * timings.beta_update / total, "%");
        printProfileLine("Update residual:", 100.0 * timings.residual_update / total, "%");
        printProfileLine("LASSO drops:", 100.0 * timings.lasso_drops / total, "%");
        printProfileLine("Update inactives:", 100.0 * timings.inactive_update / total, "%");
        printProfileLine("Update correlations:", 100.0 * timings.corr_update / total, "%");

        std::cout << "================================\n";
        std::cout << "Cycling statistics:\n";
        std::cout << "  Additions: " << num_additions_ << "\n";
        std::cout << "  Removals: " << num_removals_ << "\n";
        std::cout << "  Cycling ratio: " << std::fixed << std::setprecision(3)
                  << getCyclingRatio() << "\n";
        std::cout << "================================\n\n";
    }
}


// ============================================================================
// Helper
// ============================================================================

double TLASSO_Solver::getCyclingRatio() const {
    if (num_additions_ == 0) { return 0.0; }

    return static_cast<double>(num_removals_) /
            static_cast<double>(num_additions_);
}


double TLASSO_Solver::computeGammaSignChange(double gamhat,
                                             std::vector<bool>& drops,
                                             const Eigen::VectorXd& w_A) const {

    // Extract current beta values for active variables
    Eigen::VectorXd beta_curr(actives_.size());
    for (std::size_t i = 0; i < actives_.size(); ++i) {
        beta_curr(i) = betaPath_(actives_[i], currentStep_ - 1);
    }

    // Find zero crossing:  z = -β / w  (step size where β + γw = 0)
    Eigen::VectorXd z = -beta_curr.array() / w_A.array();

    // Find minimum positive z (first zero-crossing)
    double zmin = std::numeric_limits<double>::max();
    bool found_crossing = false;

    for (std::size_t i = 0; i < actives_.size(); ++i) {
        if (z(i) > eps_ && z(i) < zmin) {
            zmin = z(i);
            found_crossing = true;
        }
    }

    if (found_crossing && zmin < gamhat) {
        // Coefficient will cross zero before next variable enters
        gamhat = zmin;

        for (std::size_t i = 0; i < actives_.size(); ++i) {
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
            R_ = downdateR(R_, i);

            // Zero out coefficient
            betaPath_(dropped_var, currentStep_) = 0.0;

            // Record negative action (removal)
            actions_.back().push_back(-static_cast<int>(dropped_var));

            // Re-add to inactives_ for cycling support
            inactives_.push_back(dropped_var);

            // Remove from active set
            actives_.erase(actives_.begin() + i);
            Sign_.erase(Sign_.begin() + i);

            // Increment removal counter
            num_removals_++;
        }
    }
}


// ============================================================================
// Serialization
// ============================================================================

TLASSO_Solver TLASSO_Solver::load(const std::string& filename,
                                  Eigen::Map<Eigen::MatrixXd>& X) {

    TLASSO_Solver tlasso;  // No-args constructor
    const std::string solver{
        TLASSO_Solver::solverTypeToString(SolverTypeLarsBased::TLASSO)};

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
            iarchive(tlasso);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << solver << "::load: Deserialization failed - " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    // 2. Validate loaded state integrity
    if (tlasso.betaPath_.rows() == 0) {
        throw std::runtime_error(
            tlasso.concatMsg(
                solver, "::load: Loaded state is corrupted (betaPath_ is empty)"
            )
        );
    }

    // 3. Reconnect with error handling
    try {
        tlasso.reconnect(X);
        tlasso.logInfo(tlasso.concatMsg(
            "[", tlasso.solverTypeToString(), "] loaded from '", filename, "'\n",
            "  - Algorithm: ", tlasso.solverTypeToString(), "\n",
            "  - Loaded at step: ", tlasso.currentStep_, "/", tlasso.maxSteps_, "\n",
            "  - Active variables: ", tlasso.actives_.size(), "\n",
            "  - Active dummies: ", tlasso.count_active_dummies_, "\n",
            "  - Matrix dimensions: ", X.rows(), " x ", X.cols(), "\n",
            "  - Status: Ready for continued execution"
        ));
    } catch (const std::exception& e) {
        throw std::runtime_error(
            tlasso.concatMsg(
                tlasso.solverTypeToString(),
                 "::load: Reconnection failed - ", e.what()
            )
        );
    }

    return tlasso;
}
