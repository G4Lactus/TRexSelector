#include "TACGP_Solver.hpp"

// ==================================================================================
// Constructor Implementation
// ==================================================================================

// Public constructor
TACGP_Solver::TACGP_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    std::size_t num_dummies,
    bool normalize,
    bool intercept,
    bool verbose)
    : TGP_Solver(X, y, num_dummies, normalize, intercept, verbose, SolverTypeOMPBased::TACGP),
      beta1_(0.0)
{
    // Initialize TACGP-specific state
    direction_prev_.resize(0);
    direction_projected_prev_.resize(0);

}


// Default constructor
TACGP_Solver::TACGP_Solver()
    : TGP_Solver(SolverTypeOMPBased::TACGP),
      beta1_(0.0)
{
    direction_prev_.resize(0);
    direction_projected_prev_.resize(0);

}


// Protected constructor
TACGP_Solver::TACGP_Solver(SolverTypeOMPBased type)
    : TGP_Solver(type),
      beta1_(0.0)
{
    direction_prev_.resize(0);
    direction_projected_prev_.resize(0);

}


// ==================================================================================
// Main executeStep method override
// ==================================================================================

void TACGP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    while (currentStep_ < maxSteps_ &&
           actives_.size() < effective_n_ &&
           (count_active_dummies_ < T_stop || !early_stop)) {

        // ==========================================================
        // STEP 1: Find max |correlation| among ALL variables
        // ==========================================================

        auto [Cmax, new_vars] = findTiedMaxCorrelations();

        if (new_vars.empty()) {
            logInfo("No variables to add; exiting.");
            break;
        }

        if (Cmax < 100 * eps_) {
            logInfo("Max |corr| approx 0; exiting.");
            break;
        }

        currentStep_++;


        // ==========================================================
        // STEP 2: Update active set (with inherited dummy tracking)
        // ==========================================================

        actions_.emplace_back(updateActiveSet(new_vars));

        if (actives_.empty()) {
            logWarning("Active set is empty; aborting.");
            break;
        }

        // ==========================================================
        // STEP 3: Compute gradient direction and step size
        // ==========================================================

        computeGradientDirection();       // d^n = g_active + beta1 * d^{n-1}
        projectDirection();               // c^n = X_active * d^n
        step_size_ = computeStepSize();   // alpha_n = <r, c> / ||c||^2

        if (std::abs(step_size_) < eps_) {
            logInfo("Step size approx 0; exiting.");
            break;
        }

        updateActiveCoefficients();       // y_active += alpha_n * d^n
        updateResiduals();                // r -= alpha_n * c^n


        // ==========================================================
        // STEP 4: Store direction for next conjugate computation
        // ==========================================================

        storePreviousDirection();       // Save d^n and c^n for next iteration

        // ==========================================================
        // STEP 5: Diagnostics and Path Storage
        // ==========================================================

        updateBetaPath();               // update betaPath_ with active_coefficients_

        double rss = r_.dot(r_);
        RSS_.emplace_back(rss);
        R2_.emplace_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();         // track dummies in active set
        DoF_.emplace_back(actives_.size() + (intercept_ ? 1 : 0));

        // ==========================================================
        // STEP 6: Update correlations for next iteration
        // ==========================================================

        updateCorrelations();        // correlations_ = X^T r

    }
}



// ==================================================================================
// ACGP Specific Methods and Overrides
// ==================================================================================

void TACGP_Solver::computeGradientDirection() {

    // Extract gradient for active variables
    std::size_t k = actives_.size();
    direction_.resize(k);

    // Extract gradient for active variables
    #pragma omp parallel for schedule(static) if (k > 100)
    for (std::size_t i = 0; i < k; ++i) {
        direction_[i] = correlations_[actives_[i]];
    }

    // Add conjugate correction if previous direction is already included
    // and active set did not changed.
    if (static_cast<std::size_t>(direction_prev_.size()) == k
        && direction_projected_prev_.size() > 0) {

        beta1_ = computeBeta1();

        // ACGP direction: d^n = g^n + beta1 * d^{n-1}
        #pragma omp parallel for schedule(static) if (k > 100)
        for (std::size_t i = 0; i < k; ++i) {
            direction_[i] += beta1_ * direction_prev_[i];
        }

    } else {
        beta1_ = 0.0; // first iteration or active set changed
    }
}


double TACGP_Solver::computeBeta1() {

    // beta1 = <X^T X d^{n-1}, g^n> / ||c^{n-1}||^2

    // Step 1: Compute X^T (X d_{n-1}) = X^T c^{n-1} for active variables
    std::size_t k = actives_.size();
    Eigen::VectorXd Xtc(k);

    #pragma omp parallel for schedule(static) if (k > 100)
    for (std::size_t i = 0; i < k; ++i) {
        Xtc[i] = X_->col(actives_[i]).dot(direction_projected_prev_);
    }

    // Step 2: Compute <X^T X d ^{n-1}, g^n> = <Xtc, gradient_active>
    double numerator = 0.0;
    #pragma omp parallel for reduction(+:numerator) if (k > 100)
    for (std::size_t i = 0; i < k; ++i) {
        numerator += Xtc[i] * correlations_[actives_[i]];
    }

    // Step 3: Denominator || c^{n-1} ||^2
    double denominator = direction_projected_prev_.squaredNorm();

    if (denominator < eps_ * eps_) {
        logWarning("||c^{n-1}||^2 ~0; falling back to pure gradient.");
        return 0.0; // no correction
    }

    double beta = - numerator / denominator;

    // Sanity check for huge beta
    if (std::abs(beta) > 1e10) {
        logWarning("Computed beta1 is huge; numerical issue.");
        return std::copysign(1e10, beta);
    }

    return beta;
}


void TACGP_Solver::storePreviousDirection() {
    // Store current direction for next iteration
    direction_prev_ = direction_;
    direction_projected_prev_ = direction_projected_;
}

// ==================================================================================
// De-/Serialization
// ==================================================================================

void TACGP_Solver::save(const std::string& filename) const {

    std::ofstream ofs(filename, std::ios::binary);

    if (!ofs.is_open()) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::save: Cannot open file '", filename,
                "' for writing. Check permissions and path."
            )
        );
    }

    try {
        cereal::PortableBinaryOutputArchive oarchive(ofs);
        oarchive(*this);

        logInfo(concatMsg(
            "[", solverTypeToString(), "] saved to '", filename, "'\n"
        ));


    } catch (const std::exception& e) {
        throw std::runtime_error(
            concatMsg(
                solverTypeToString(),
                "::save: Serialization to file '", filename,
                "' failed with error: ", e.what()
            )
        );
    }
}


TACGP_Solver TACGP_Solver::load(const std::string& filename, Eigen::Map<Eigen::MatrixXd>& X) {

    TACGP_Solver tacgp; // Uses no-arg constructor
    const std::string solver{TACGP_Solver::solverTypeToString(SolverTypeOMPBased::TACGP)};

    // 1. Deserialize from file with error handling
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            std::ostringstream oss;
            oss << solver << "::load: Cannot open file '" << filename;
            throw std::runtime_error(oss.str());
        }

        try {
            cereal::PortableBinaryInputArchive iarchive(ifs);
            iarchive(tacgp);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << solver << "::load: Deserialization failed - " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    // 2. Validate loaded state integrity
    if (tacgp.getBetaPath().rows() != X.cols()) {
        throw std::runtime_error(
            tacgp.concatMsg(
                solver, "::load: Loaded solver design matrix columns (",
                tacgp.getBetaPath().rows(), ") do not match X columns (",
                X.cols(), "). Ensure X matches the original training matrix."
            )
        );
    }

    // 3. Reconnect with error handling
    try {
        tacgp.reconnect(X);

        tacgp.logInfo(tacgp.concatMsg(
            "[", tacgp.solverTypeToString(), "] loaded from '", filename, "'\n",
            "  - Algorithm: ", tacgp.solverTypeToString(), "\n",
            "  - Loaded at step: ", tacgp.currentStep_, "/",
            "  - Active variables: ", tacgp.actives_.size(), "\n",
            "  - Matrix dimensions: ", X.rows(), " x ", X.cols(), "\n",
            "  - Status: Ready for continued execution"
        ));
    } catch (const std::exception& e) {
        throw std::runtime_error(
            tacgp.concatMsg(
                tacgp.solverTypeToString(), "::load: Reconnection failed - ", e.what()
            )
        );
    }

    return tacgp;
}
