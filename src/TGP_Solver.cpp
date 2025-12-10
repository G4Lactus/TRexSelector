# include "TGP_Solver.hpp"

// ==================================================================================
// Constructor Implementation
// ==================================================================================

// Public constructor
TGP_Solver::TGP_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    std::size_t num_dummies,
    bool normalize,
    bool intercept,
    bool verbose)
    : TOMP_Solver(X, y, num_dummies, normalize, intercept, verbose, SolverType::TGP),
      step_size_(0.0) {

    // Initialize active_coefficients_ (empty at start)
    active_coefficients_.resize(0);
}

// Protected constructor
TGP_Solver::TGP_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    std::size_t num_dummies,
    bool normalize,
    bool intercept,
    bool verbose,
    SolverType type)
    : TOMP_Solver(X, y, num_dummies, normalize, intercept, verbose, type),
      step_size_(0.0)
{
    active_coefficients_.resize(0);
}

// Default constructor
TGP_Solver::TGP_Solver()
    : TOMP_Solver(SolverType::TGP),
      step_size_(0.0)
{
    active_coefficients_.resize(0);
}

// Protected default constructor
TGP_Solver::TGP_Solver(SolverType type)
    : TOMP_Solver(type), step_size_(0.0)
{
    active_coefficients_.resize(0);
}


// ==================================================================================
// Main Algorithm: executeStep()
// ==================================================================================

void TGP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
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

        computeGradientDirection();      // d_n = g_active
        projectDirection();              // c_n = X_active * d_n
        step_size_ = computeStepSize();  // alpha_n = <r, c> / ||c||^2

        if (std::abs(step_size_) < eps_) {
            logInfo("Step size approx 0; exiting.");
            break;
        }

        updateActiveCoefficients();      // y_active += alpha_n * d_n
        updateResiduals();               // r -= alpha_n * c_n

        // ==========================================================
        // STEP 4: Update path and diagnostics
        // ==========================================================

        updateBetaPath(); // update betaPath_ with active_coefficients_

        double rss = r_.dot(r_);
        RSS_.emplace_back(rss);
        R2_.emplace_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();         // track dummies in active set
        DoF_.emplace_back(actives_.size() + (intercept_ ? 1 : 0));

        // ==========================================================
        // STEP 5: Update inactive set and correlations
        // ==========================================================
        computeCorrelations();
    }
}


// ==================================================================================
// GP-Specific Helper Methods
// ==================================================================================

void TGP_Solver::computeGradientDirection() {

    // Gradient for active variables: d_n = g_active = X_active^T r
    std::size_t k = actives_.size();
    direction_.resize(k);

    #pragma omp parallel for schedule(static) if (k > 100)
    for (std::size_t i = 0; i < k; ++i) {
        direction_[i] = correlations_[actives_[i]];
    }
}


void TGP_Solver::projectDirection() {

    // Projected direction into data space: c_n = X_active * d_n
    std::size_t n = X_->rows();
    direction_projected_.resize(n);
    direction_projected_.setZero();

    // c = X_active * d (manual loop for memory efficiency in p >> n)
    for (std::size_t i = 0; i < actives_.size(); ++i) {
        direction_projected_ += direction_[i] * X_->col(actives_[i]);
    }
}


double TGP_Solver::computeStepSize() {

    // Step size: alpha_n = <r, c> / ||c||^2
    double numerator = r_.dot(direction_projected_);
    double denominator = direction_projected_.squaredNorm();

    if (denominator < eps_ * eps_) {
        logWarning("||c_n||^2 ~0; numerical issue.");
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
        active_coefficients_.conservativeResize(k);
    }

    // Directional update: y_active += alpha_n * d_n
    #pragma omp parallel for schedule(static) if (k > 100)
    for (std::size_t i = 0; i < k; ++i) {
        active_coefficients_[i] += step_size_ * direction_[i];
    }
}


void TGP_Solver::updateResiduals() {
    // GP residual update: r -= alpha_n * c_n
    r_.noalias() -= step_size_ * direction_projected_;
}


void TGP_Solver::updateBetaPath() {

    if (currentStep_ >= static_cast<std::size_t>(betaPath_.cols())) {
        betaPath_.conservativeResize(X_->cols(), currentStep_ + 1);
    }

    // Create full coefficient vector (inactives = 0)
    Eigen::VectorXd beta_hat = Eigen::VectorXd::Zero(X_->cols());

    // Copy from active_coefficients_
    std::size_t k = actives_.size();
    #pragma omp parallel for schedule(static) if (k > 100)
    for (std::size_t i = 0; i < k; ++i) {
        beta_hat[actives_[i]] = active_coefficients_[i];
    }

    betaPath_.col(currentStep_) = beta_hat;
}


// ==================================================================================
// Overrides
// ==================================================================================

std::pair<double, std::vector<std::size_t>> TGP_Solver::findTiedMaxCorrelations() const {
    double Cmax = 0.0;
    std::vector<std::size_t> tied;
    tied.reserve(10);

    // GP searches ALL variables (active + inactive)
    std::unordered_set<std::size_t> dropped_set(
        dropped_indices_.begin(), dropped_indices_.end()
    );

    for (std::size_t j = 0; j < static_cast<std::size_t>(X_->cols()); ++j) {

        // Skip dropped variables (collinear) variables
        if (dropped_set.find(j) != dropped_set.end()) {
            continue;
        }

        double abs_corr = std::abs(correlations_[j]);

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
        bool already_active = (std::find(actives_.begin(), actives_.end(), j_new)
                               != actives_.end());

        if (already_active) {
            // Re-selected atom: just record
            actions_this_step.push_back(static_cast<int>(j_new));
        } else {
            // New atom: add directly
            actives_.push_back(j_new);
            actions_this_step.push_back(static_cast<int>(j_new));
            num_additions_++;

            // Initialize coefficient
            active_coefficients_.conservativeResize(actives_.size());
            active_coefficients_(actives_.size() - 1) = 0.0;

            // Track dummy entry
            if (j_new >= dummy_start_idx_) {
                count_active_dummies_++;
            }
        }
    }

    return actions_this_step;
}


void TGP_Solver::computeCorrelations() {
    // GP needs correlations for all variables
    std::unordered_set<std::size_t> dropped_set(
        dropped_indices_.begin(), dropped_indices_.end()
    );

    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < static_cast<std::size_t>(X_->cols()); ++j) {
        if (dropped_set.find(j) == dropped_set.end()) {
            correlations_[j] = X_->col(j).dot(r_);
        } else {
            correlations_[j] = 0.0;
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


TGP_Solver TGP_Solver::load(const std::string& filename, Eigen::Map<Eigen::MatrixXd>& X) {

    TGP_Solver tgp; // Uses no-arg constructor
    const std::string solver{TGP_Solver::solverTypeToString(SolverType::TGP)};

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
            iarchive(tgp);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << solver << "::load: Deserialization failed - " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    // 2. Validate loaded state integrity
    if (tgp.getBetaPath().rows() != X.cols()) {
        throw std::runtime_error(
            tgp.concatMsg(
                solver, "::load: Loaded solver design matrix columns (",
                tgp.getBetaPath().rows(), ") do not match X columns (",
                X.cols(), "). Ensure X matches the original training matrix."
            )
        );
    }

    // 3. Reconnect with error handling
    try {
        tgp.reconnect(X);

        tgp.logInfo(tgp.concatMsg(
            "[", tgp.solverTypeToString(), "] loaded from '", filename, "'\n",
            "  - Algorithm: ", tgp.solverTypeToString(), "\n",
            "  - Loaded at step: ", tgp.currentStep_, "/",
            "  - Active variables: ", tgp.actives_.size(), "\n",
            "  - Matrix dimensions: ", X.rows(), " x ", X.cols(), "\n",
            "  - Status: Ready for continued execution"
        ));
    } catch (const std::exception& e) {
        throw std::runtime_error(
            tgp.concatMsg(
                tgp.solverTypeToString(), "::load: Reconnection failed - ", e.what()
            )
        );
    }

    return tgp;
}
