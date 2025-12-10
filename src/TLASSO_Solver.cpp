
#include "TLASSO_Solver.hpp"


// ============================================================================
// Core executeStep() Implementation
// ============================================================================

void TLASSO_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    while (currentStep_ < maxSteps_ &&
           !inactives_.empty() &&
           actives_.size() < effective_n_ &&
           (count_active_dummies_ < T_stop || !early_stop)) {

        // ========================================================
        // STEP 1: Extract correlations for inactives
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
        // STEP 3: Active set update (with inherited dummy tracking)
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
        // STEP 5: T-LASSO modification - check for sign changes
        // ========================================================

        std::vector<bool> drops(actives_.size(), false);
        double gamma_entry = computeStepSize(Cmax, u);
        double gamma = computeGammaSignChange(gamma_entry, drops, w_A);

        // ========================================================
        // Update Beta
        // ========================================================

        updateBetaPath(w_A, gamma);

        // ========================================================
        // Update Residuals
        // ========================================================

        r_ -= gamma * u;

        // ========================================================
        // Update diagnostics
        // ========================================================

        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        // ========================================================
        // Handle T-LASSO drops
        // ========================================================

        processLassoDrops(drops);

        // T-LASSO: Record dummy count and DoF after all adds/removes
        dummies_at_step_.push_back(count_active_dummies_);
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));

        // ========================================================
        // Update inactive set
        // ========================================================

        updateInactiveSet();

        // ========================================================
        // STEP 9: Recompute correlations for next iteration
        // ========================================================

        computeCorrelations();
    }
}


// ============================================================================
// Internal Helper
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
            // Remove from active set
            actives_.erase(actives_.begin() + i);
            Sign_.erase(Sign_.begin() + i);
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
        TLASSO_Solver::solverTypeToString(SolverType::TLASSO)};

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
            "[", tlasso.solverTypeToString(), "] loaded from '",
             filename, "'\n",
            "  - Algorithm: ", tlasso.solverTypeToString(), "\n",
            "  - Loaded at step: ", tlasso.currentStep_, "/",
             tlasso.maxSteps_, "\n",
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
