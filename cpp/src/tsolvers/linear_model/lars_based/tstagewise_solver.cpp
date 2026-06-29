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
#include <tsolvers/linear_model/lars_based/tstagewise_solver.hpp>

// ============================================================================

// Embed into namespace trex::tsolvers::linear_model::lars_based
namespace trex::tsolvers::linear_model::lars_based {

// ============================================================================
// Constructors
// ============================================================================

TSTAGEWISE_Solver::TSTAGEWISE_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D,
    Eigen::Map<Eigen::VectorXd>& y,
    bool normalize,
    bool intercept,
    bool verbose,
    ScalingMode scaling_mode
)
    : TLARS_Solver(X, D, y, normalize, intercept, verbose,
                    SolverTypeLarsBased::TSTAGEWISE, scaling_mode),
      num_removals_(0) {
}

// ============================================================================
// Core executeStep()
// ============================================================================

void TSTAGEWISE_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();
    Eigen::setNbThreads(1);

    while (
        currentStep_ < maxSteps_ &&
        !inactives_.empty() &&
        actives_.size() < effective_n_ &&
        (count_active_dummies_ < T_stop || !early_stop)
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
            break;
        }

        // ========================================================
        // Step 3: T-Stagewise NNLS Direction Check Loop
        // ========================================================
        Eigen::VectorXd w_A = computeStagewiseDirection();

        if (actives_.empty()) {
            logWarning("Active set is empty after NNLS reduction; aborting.");
            break;
        }

        // ========================================================
        // Step 4: Update Inactives
        // ========================================================
        // If NNLS dropped variables, they must rejoin inactives_
        // before computeStepSize
        updateInactiveSet();

        // ========================================================
        // Step 5: Equi-angular projection and Step Size
        // ========================================================
        Eigen::VectorXd u = equiangularVector(w_A);
        auto [gamma, a] = computeStepSize(Cmax, u);

        // ========================================================
        // Step 6: Update Beta Path and Residuals
        // ========================================================
        updateBetaPath(w_A, gamma);
        r_ -= gamma * u;

        // ========================================================
        // Step 7: Update Correlations
        // ========================================================
        updateCorrelations(gamma, a);

        // ========================================================
        // Step 8: Update diagnostics and dummy tracking
        // ========================================================
        double rss = r_.dot(r_);
        RSS_.push_back(rss);
        R2_.push_back(1.0 - rss / RSS_[0]);

        updateDummyTracking();
        DoF_.push_back(actives_.size() + (intercept_ ? 1 : 0));
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

    // 2. Update dummy tracking
    if (dropped_var >= dummy_start_idx_ && count_active_dummies_ > 0) {
        count_active_dummies_--;
        logInfo(concatMsg("Dummy variable ", dropped_var,
                          " removed during NNLS reduction (count: ",
                          count_active_dummies_, ")\n"));
    }

    // 3. Downdate Cholesky factor R_
    R_ = downdateR(R_, drop_idx, eps_);

    // 4. Record the removal action in current step
    // Negative index indicates removal
    actions_.back().push_back(-static_cast<int>(dropped_var));

    // 5. Remove from active set and signs
    actives_.erase(actives_.begin() + drop_idx);
    Sign_.erase(Sign_.begin() + drop_idx);

    // 6. Mark for inactive set rebuild at the end of the step
    any_dropped_ = true;
    num_removals_++;
}

// ============================================================================
// Internal Helpers & Serialization
// ============================================================================

double TSTAGEWISE_Solver::getCyclingRatio() const {
    if (num_additions_ == 0) { return 0.0; }
    return static_cast<double>(num_removals_) / static_cast<double>(num_additions_);
}

// ============================================================================
// De-/Serialization
// ============================================================================

void TSTAGEWISE_Solver::save(const std::string& filename) const {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error(
            concatMsg(solverTypeToString(), "::save: Cannot open '", filename, "'")
        );
    }
    cereal::PortableBinaryOutputArchive oarchive(os);
    oarchive(*this);
    logInfo(concatMsg("[", solverTypeToString(), "] saved to '", filename, "'\n"));
}


TSTAGEWISE_Solver TSTAGEWISE_Solver::load(const std::string& filename,
                                          Eigen::Map<Eigen::MatrixXd>& X,
                                          Eigen::Map<Eigen::MatrixXd>& D) {
    TSTAGEWISE_Solver solver;
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error(
            solver.concatMsg(solver.solverTypeToString(), "::load: Cannot open '", filename, "'")
        );
    }
    cereal::PortableBinaryInputArchive iarchive(is);
    iarchive(solver);
    solver.reconnect(X, D);
    return solver;
}

// ============================================================================
} /* End of namespace trex::tsolvers::linear_model::lars_based */
