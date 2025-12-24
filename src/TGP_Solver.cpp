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
    : TOMP_Solver(X, y, num_dummies, normalize, intercept, verbose, SolverTypeOMPBased::TGP),
      step_size_(0.0) {
}

// Protected constructor
TGP_Solver::TGP_Solver(
    Eigen::Map<Eigen::MatrixXd>& X,
    Eigen::Map<Eigen::VectorXd>& y,
    std::size_t num_dummies,
    bool normalize,
    bool intercept,
    bool verbose,
    SolverTypeOMPBased type)
    : TOMP_Solver(X, y, num_dummies, normalize, intercept, verbose, type),
      step_size_(0.0) {}

// Default constructor
TGP_Solver::TGP_Solver()
    : TOMP_Solver(SolverTypeOMPBased::TGP),
      step_size_(0.0) {}

// Protected default constructor
TGP_Solver::TGP_Solver(SolverTypeOMPBased type)
    : TOMP_Solver(type), step_size_(0.0) {}



// ==================================================================================
// Main Algorithm: executeStep()
// ==================================================================================

void TGP_Solver::executeStep(std::size_t T_stop, bool early_stop) {
    validateConnected();

    struct StepTimings {
        double find_max_corr = 0.0;
        double update_active_set = 0.0;
        double gradient_computation = 0.0;
        double step_size_computation = 0.0;
        double coeff_update = 0.0;
        double residual_update = 0.0;
        double corr_update = 0.0;
        double beta_update = 0.0;
        std::size_t num_steps = 0;
    } timings;

    while (currentStep_ < maxSteps_ &&
           actives_.size() < effective_n_ &&
           (count_active_dummies_ < T_stop || !early_stop)) {


        // ==========================================================
        // STEP 1: Find max |correlation| among ALL variables
        // ==========================================================
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
        currentStep_++;

        // ==========================================================
        // STEP 2: Update active set (with inherited dummy tracking)
        // ==========================================================
        t1 = std::chrono::high_resolution_clock::now();
        actions_.emplace_back(updateActiveSet(new_vars));
        t2 = std::chrono::high_resolution_clock::now();
        timings.update_active_set += std::chrono::duration<double, std::milli>(t2 - t1).count();

        if (actives_.empty()) {
            logWarning("Active set is empty; aborting.");
            break;
        }

        // ==========================================================
        // STEP 3: Compute gradient direction and step size
        // ==========================================================
        t1 = std::chrono::high_resolution_clock::now();
        computeGradientDirection();      // d_n = g_active
        t2 = std::chrono::high_resolution_clock::now();
        timings.gradient_computation += std::chrono::duration<double, std::milli>(t2 - t1).count();

        t1 = std::chrono::high_resolution_clock::now();
        projectDirection();              // c_n = X_active * d_n
        t2 = std::chrono::high_resolution_clock::now();
        timings.gradient_computation += std::chrono::duration<double, std::milli>(t2 - t1).count();

        t1 = std::chrono::high_resolution_clock::now();
        step_size_ = computeStepSize();  // alpha_n = <r, c> / ||c||^2
        t2 = std::chrono::high_resolution_clock::now();
        timings.step_size_computation += std::chrono::duration<double, std::milli>(t2 - t1).count();

        if (std::abs(step_size_) < eps_) {
            logInfo("Step size approx 0; exiting.");
            break;
        }

        t1 = std::chrono::high_resolution_clock::now();
        updateActiveCoefficients();      // y_active += alpha_n * d_n
        t2 = std::chrono::high_resolution_clock::now();
        timings.coeff_update += std::chrono::duration<double, std::milli>(t2 - t1).count();

        t1 = std::chrono::high_resolution_clock::now();
        updateResiduals();               // r -= alpha_n * c_n
        t2 = std::chrono::high_resolution_clock::now();
        timings.residual_update += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // ==========================================================
        // STEP 4: Update path and diagnostics
        // ==========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateBetaPath(); // update betaPath_ with active_coefficients_
        t2 = std::chrono::high_resolution_clock::now();
        timings.beta_update += std::chrono::duration<double, std::milli>(t2 - t1).count();

        double rss = r_.dot(r_);
        RSS_.emplace_back(rss);
        R2_.emplace_back(1.0 - rss / RSS_[0]);
        updateDummyTracking();         // track dummies in active set
        DoF_.emplace_back(actives_.size() + (intercept_ ? 1 : 0));

        // ==========================================================
        // STEP 5: Update correlations for ALL variables
        // ==========================================================
        t1 = std::chrono::high_resolution_clock::now();
        updateCorrelations();
        t2 = std::chrono::high_resolution_clock::now();
        timings.corr_update += std::chrono::duration<double, std::milli>(t2 - t1).count();

        timings.num_steps++;
    }

    if (verbose_ && timings.num_steps > 0) {
        std::cout << "\n=== T-GP Profiling Summary ===\n";
        std::cout << "Total steps: " << timings.num_steps << "\n";

        // Time per step computations
        std::cout << "Time per step (avg):\n" << std::fixed << std::setprecision(1);
        printProfileLine("Find max corr:", timings.find_max_corr / timings.num_steps, " ms");
        printProfileLine("Update active:", timings.update_active_set / timings.num_steps, " ms");
        printProfileLine("Gradient computation:",
                         timings.gradient_computation / timings.num_steps, " ms");
        printProfileLine("Step size computation:",
                         timings.step_size_computation / timings.num_steps, " ms");
        printProfileLine("Coeff update:", timings.coeff_update / timings.num_steps, " ms");
        printProfileLine("Residual update:", timings.residual_update / timings.num_steps, " ms");
        printProfileLine("Update beta:", timings.beta_update / timings.num_steps, " ms");
        printProfileLine("Update correlations:", timings.corr_update / timings.num_steps, " ms");

        // Total time breakdown
        double total = timings.find_max_corr + timings.update_active_set +
                    timings.gradient_computation + timings.step_size_computation +
                    timings.coeff_update + timings.residual_update +
                    timings.beta_update + timings.corr_update;

        // Percentage breakdown
        std::cout << "\nPercentage breakdown:\n";
        printProfileLine("Find max corr:", 100.0 * timings.find_max_corr / total, "%");
        printProfileLine("Update active:", 100.0 * timings.update_active_set / total, "%");
        printProfileLine("Gradient computation:",
                         100.0 * timings.gradient_computation / total, "%");
        printProfileLine("Step size computation:",
                         100.0 * timings.step_size_computation / total, "%");
        printProfileLine("Coeff update:", 100.0 * timings.coeff_update / total, "%");
        printProfileLine("Residual update:", 100.0 * timings.residual_update / total, "%");
        printProfileLine("Update beta:", 100.0 * timings.beta_update / total, "%");
        printProfileLine("Update correlations:", 100.0 * timings.corr_update / total, "%");
        std::cout << "===================================\n\n";
    }
}


// ==================================================================================
// GP-Specific Helper Methods
// ==================================================================================

void TGP_Solver::computeGradientDirection() {

    // Gradient for active variables: d_n = g_active == X_active^T r
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

    if (denominator < eps_ * eps_) { // implicit collinearity check
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
    std::size_t p = static_cast<std::size_t>(X_->cols());

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
        bool already_active = (std::find(actives_.begin(),
                               actives_.end(), j_new)
                               != actives_.end());

        actions_this_step.push_back(static_cast<int>(j_new));

        if (!already_active) {
            // New atom: add directly
            actives_.push_back(j_new);
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


void TGP_Solver::updateCorrelations() {
    // GP needs correlations for all variables
    std::size_t p = static_cast<std::size_t>(X_->cols());
    std::size_t num_dropped = dropped_indices_.size();

    if (num_dropped == 0) { // No dropped variables: simple update
        #pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < p; ++j) {
            correlations_[j] = X_->col(j).dot(r_);
        }
        return;
    } else if (num_dropped > 500) {  // Fallback for unusual cases
        std::unordered_set<std::size_t> dropped_set(
            dropped_indices_.begin(), dropped_indices_.end());
        #pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < p; ++j) {
            correlations_[j] = dropped_set.contains(j) ? 0.0 : X_->col(j).dot(r_);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < p; ++j) {
            bool is_dropped = false;
            for (auto idx : dropped_indices_) {
                if (idx == j) { is_dropped = true; break; }
            }
            correlations_[j] = is_dropped ? 0.0 : X_->col(j).dot(r_);
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
    const std::string solver{TGP_Solver::solverTypeToString(SolverTypeOMPBased::TGP)};

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
