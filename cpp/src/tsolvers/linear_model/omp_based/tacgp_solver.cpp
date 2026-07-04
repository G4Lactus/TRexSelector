// ==================================================================================
// tacgp_solver.cpp
// ==================================================================================
/**
 * @file tacgp_solver.cpp
 *
 * @brief Implementation of the Terminating Approximate Conjugate Gradient Pursuit
 * (T-ACGP) solver.
 */
 // ==================================================================================

// std includes
#include <algorithm>
#include <fstream>
#include <stdexcept>

// tsolvers includes
#include <tsolvers/linear_model/omp_based/tacgp_solver.hpp>

// ==================================================================================

// Embedded into trex::tsolvers::linear_model::omp_based namespace
namespace trex::tsolvers::linear_model::omp_based {

// ==================================================================================
// Constructors
// ==================================================================================

TACGP_Solver::TACGP_Solver(SolverTypeOMPBased solver_type)
    : TGP_Solver(solver_type) {}


// ==================================================================================
// Main executeStep method override
// ==================================================================================

void TACGP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();
    Eigen::setNbThreads(1);

    while (currentStep_ < maxSteps_ &&
           actives_.size() < effective_n_ &&
           (count_active_dummies_ < T_stop || !early_stop)) {

        // =================================================================
        // STEP 1: Find max correlation |c^{n}| among ALL variables
        // =================================================================

        auto [Cmax, new_vars] = findTiedMaxCorrelations();
        if (new_vars.empty() || Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0 or no variables to add; exiting.");
            break;
        }

        pruneTiedDummies(new_vars, T_stop, early_stop);

        // ================================================================
        // STEP 2: Update active set (with inherited dummy tracking)
        // ================================================================

        actions_.emplace_back(updateActiveSet(new_vars));
        if (actives_.empty()) {
            logWarning("Active set is empty; aborting.");
            break;
        }

        // ================================================================
        // STEP 3 & 4: Approximate conjugate direction
        //   d^{n} = c^{n}_A + b_1 * d^{n-1}
        // ================================================================

        computeGradientDirection();

        // ================================================================
        // STEP 5: Project direction into data space
        // p^{n} = X_A * d^{n}
        // ================================================================

        projectDirection();

        // ================================================================
        // STEP 6: Store d^n and p^n as d^{n-1} and p^{n-1}
        // ================================================================

        storePreviousDirection();

        // ================================================================
        // STEP 7: Compute step size
        //  alpha^{n} =  <r^{n-1}, p^{n}> / ||p^{n}||^2
        // ================================================================

        step_size_ = computeStepSize();
        if (std::abs(step_size_) < eps_) {
            logInfo("Step size approx 0; exiting.");
            break;
        }

        // ==========================================================
        // STEP 8: Update active coefficients:
        // beta_A^n += alpha^n * d^n
        // ==========================================================

        updateActiveCoefficients();

        // ================================================================
        // STEP 9: Update residual
        // r^{n} = r^{n-1} - alpha^{n} * p^{n}
        // ================================================================

        updateResiduals();

        // ================================================================
        // STEP 10: Update path and diagnostics
        // ================================================================

        currentStep_++;
        updateBetaPath();

        double rss = r_.dot(r_);
        RSS_.emplace_back(rss);
        R2_.emplace_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();
        DoF_.emplace_back(actives_.size() + (intercept_ ? 1 : 0));

        // ================================================================
        // STEP 11: Update correlations c^n = X^T r for ALL variables
        // ================================================================

        updateCorrelations();
    }
}


// ==================================================================================
// ACGP Specific Methods and Overrides
// ==================================================================================

void TACGP_Solver::computeGradientDirection() {

    // Extract active correlations c^{n}_A
    std::size_t k = actives_.size();
    direction_.resize(static_cast<Eigen::Index>(k));

    // Extract gradient for active variables
    #pragma omp parallel for schedule(static) if (k > 100)
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(k); ++i) {
        direction_[i] = correlations_[static_cast<Eigen::Index>(actives_[i])];
    }

    // Add conjugate correction if previous direction is already included
    // and active set did not changed since last iteration
    if (static_cast<std::size_t>(direction_prev_.size()) == k
        && projected_direction_prev_.size() > 0) {

        b_1_ = computeB_1();

        // d^n = c^n_A + b_1 * d^{n-1}
        #pragma omp parallel for schedule(static) if (k > 100)
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(k); ++i) {
            direction_[i] += b_1_ * direction_prev_[i];
        }

    } else {
        b_1_ = 0.0; // first iteration or active set changed
    }
}


double TACGP_Solver::computeB_1() {

    // b_1 = - <p^{n-1}, X_A c^n_A> / ||p^{n-1}||^2
    std::size_t k = actives_.size();

    // Step 1: X_A^T p^{n-1}  (one dot product per active column, cost O(nk))
    Eigen::VectorXd Xtp(k);

    #pragma omp parallel for schedule(static) if (k > 100)
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(k); ++i) {
        Xtp[i] = getColumn(actives_[i]).dot(projected_direction_prev_);
    }

    // Step 2: Numerator = <c^n_A, X_A^T p^{n-1}>
    double numerator = 0.0;
    #pragma omp parallel for reduction(+:numerator) if (k > 100)
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(k); ++i) {
        numerator += Xtp[i] * correlations_[static_cast<Eigen::Index>(actives_[i])];
    }

    // Step 3: Denominator = ||p^{n-1}||^2
    double denominator = projected_direction_prev_.squaredNorm();

    if (denominator < eps_) {
        logWarning("||p^{n-1}||^2 ~0; falling back to pure gradient.");
        return 0.0; // no correction
    }

    // Combine with negative sign
    double b_1 = - numerator / denominator;

    // Sanity check for huge b_1
    if (std::abs(b_1) > 1e10) {
        logWarning("Computed b_1 is huge; numerical issue.");
        return std::copysign(1e10, b_1);
    }

    return b_1;
}


void TACGP_Solver::storePreviousDirection() {
    // Store current direction and projection for next iteration
    // d^{n-1} <- d^{n}
    direction_prev_ = direction_;
    // p^{n-1} <- p^{n}
    projected_direction_prev_ = projected_direction_;
}


// ==================================================================================
// De-/Serialization
// ==================================================================================

void TACGP_Solver::save(const std::string& filename) const {

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


TACGP_Solver TACGP_Solver::load(
    const std::string& filename,
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::MatrixXd>& D
) {

    TACGP_Solver tacgp;
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error(
            tacgp.concatMsg(tacgp.solverTypeToString(), "::load: Cannot open '", filename, "'")
        );
    }

    cereal::PortableBinaryInputArchive iarchive(is);
    iarchive(tacgp);

    tacgp.reconnect(X, D);
    return tacgp;
}

// ==================================================================================
} /* End of namespace trex::tsolvers::linear_model::omp_based */
